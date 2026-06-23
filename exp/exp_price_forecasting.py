from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')
plt.switch_backend('agg')


DEFAULT_LOSS_PARAMS = {
    'high_threshold': 300.0,
    'low_threshold': 50.0,
    'delta': 1.0,
    'extreme_weight': 2.0,
    'high_under_weight': 2.5,
    'low_over_weight': 2.5,
    'max_dynamic_weight': 5.0,
    'change_loss_weight': 0.2,
}


class WeightedHuberLoss(nn.Module):
    def __init__(self, high_threshold, low_threshold, delta,
                 extreme_weight, high_under_weight, low_over_weight,
                 max_dynamic_weight, change_loss_weight,
                 price_mean=None, price_scale=None):
        super(WeightedHuberLoss, self).__init__()
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.delta = delta
        self.extreme_weight = extreme_weight
        self.high_under_weight = high_under_weight
        self.low_over_weight = low_over_weight
        self.max_dynamic_weight = max_dynamic_weight
        self.change_loss_weight = change_loss_weight
        self.price_mean = price_mean
        self.price_scale = price_scale

    def _huber(self, pred, true):
        error = pred - true
        abs_error = torch.abs(error)
        quadratic = torch.minimum(abs_error, torch.tensor(self.delta, device=pred.device, dtype=pred.dtype))
        linear = abs_error - quadratic
        return 0.5 * quadratic * quadratic + self.delta * linear

    def _to_real_price(self, value):
        if self.price_mean is None or self.price_scale is None:
            return value
        mean = torch.as_tensor(self.price_mean, device=value.device, dtype=value.dtype)
        scale = torch.as_tensor(self.price_scale, device=value.device, dtype=value.dtype)
        return value * scale + mean

    def _dynamic_weight(self, pred, true):
        true_real = self._to_real_price(true)
        high_mask = true_real >= self.high_threshold
        low_mask = true_real <= self.low_threshold
        high_under_mask = high_mask & (pred < true)
        low_over_mask = low_mask & (pred > true)

        weights = torch.ones_like(true)
        weights = torch.where(
            high_mask | low_mask,
            weights * self.extreme_weight,
            weights,
        )
        weights = torch.where(
            high_under_mask,
            weights * self.high_under_weight,
            weights,
        )
        weights = torch.where(
            low_over_mask,
            weights * self.low_over_weight,
            weights,
        )
        return torch.clamp(weights, max=self.max_dynamic_weight)

    def forward(self, pred, true):
        weights = self._dynamic_weight(pred, true)
        level_loss = (self._huber(pred, true) * weights).mean()

        if pred.size(1) > 1:
            pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
            true_diff = true[:, 1:, :] - true[:, :-1, :]
            change_loss = self._huber(pred_diff, true_diff).mean()
        else:
            change_loss = pred.new_tensor(0.0)
        return level_loss + self.change_loss_weight * change_loss


class Exp_Price_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Price_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)

    def _select_criterion(self, price_scaler=None):
        price_mean = None
        price_scale = None
        if price_scaler is not None:
            price_mean = float(price_scaler.mean_[0])
            price_scale = float(price_scaler.scale_[0])
        return WeightedHuberLoss(
            **DEFAULT_LOSS_PARAMS,
            price_mean=price_mean,
            price_scale=price_scale,
        )

    def _move_batch_to_device(self, batch):
        moved = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved[key] = value.float().to(self.device)
            else:
                moved[key] = value
        return moved

    @staticmethod
    def _unwrap_outputs(outputs):
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch in vali_loader:
                batch = self._move_batch_to_device(batch)
                outputs = self._unwrap_outputs(self.model(batch))
                loss = criterion(outputs, batch['price_future'])
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion(getattr(train_data, 'price_scaler', None))

        scheduler = lr_scheduler.ReduceLROnPlateau(
            model_optim,
            mode='min',
            factor=0.5,
            patience=3,
        )

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        loss_history = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch = self._move_batch_to_device(batch)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._unwrap_outputs(self.model(batch))
                        loss = criterion(outputs, batch['price_future'])
                    train_loss.append(loss.item())
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    outputs = self._unwrap_outputs(self.model(batch))
                    loss = criterion(outputs, batch['price_future'])
                    train_loss.append(loss.item())
                    loss.backward()
                    model_optim.step()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Train EarlyStop Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, train_loss, test_loss))
            loss_history.append([epoch + 1, train_loss, train_loss, test_loss])
            early_stopping(train_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj == 'TST':
                scheduler.step(train_loss)
                print('Updating learning rate to {}'.format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        self._save_loss_history(setting, loss_history)
        return self.model

    def _save_loss_history(self, setting, loss_history):
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(folder_path + 'loss.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'early_stop_loss', 'test_loss'])
            writer.writerows(loss_history)
        if loss_history:
            with open(folder_path + 'last_loss.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['metric', 'value'])
                writer.writerow(['epoch', loss_history[-1][0]])
                writer.writerow(['train_loss', loss_history[-1][1]])
                writer.writerow(['early_stop_loss', loss_history[-1][2]])
                writer.writerow(['test_loss', loss_history[-1][3]])

    def _test_future_datetimes(self, test_data, sample_count):
        datetimes = []
        for start in test_data.valid_starts[:sample_count]:
            hist_end = start + self.args.seq_len
            future_end = hist_end + self.args.pred_len
            datetimes.append(test_data.datetime.iloc[hist_end:future_end].to_numpy())
        return np.asarray(datetimes)

    @staticmethod
    def _save_prediction_table(folder_path, datetimes, pred_real, true_real):
        rows = []
        for sample_idx in range(pred_real.shape[0]):
            for step_idx in range(pred_real.shape[1]):
                true_value = float(true_real[sample_idx, step_idx, 0])
                pred_value = float(pred_real[sample_idx, step_idx, 0])
                rows.append({
                    'sample': sample_idx,
                    'horizon': step_idx + 1,
                    'datetime': pd.Timestamp(datetimes[sample_idx, step_idx]),
                    'true': true_value,
                    'pred': pred_value,
                    'error': pred_value - true_value,
                })
        pd.DataFrame(rows).to_csv(folder_path + 'predictions_with_datetime.csv', index=False)

    @staticmethod
    def _save_prediction_plot(folder_path, datetimes, pred_real, true_real, sample_idx=0):
        plot_times = pd.to_datetime(datetimes[sample_idx])
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(plot_times, true_real[sample_idx, :, 0], label='GroundTruth', linewidth=2)
        ax.plot(plot_times, pred_real[sample_idx, :, 0], label='Prediction', linewidth=2)
        ax.set_title('Price Forecast {}'.format(plot_times[0].strftime('%Y-%m-%d %H:%M')))
        ax.set_xlabel('Datetime')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(folder_path + 'prediction_with_datetime.png', dpi=200)
        plt.close(fig)

    def _metric_config_rows(self):
        args = self.args
        rows = [
            [],
            ['config', 'value'],
            ['task_name', args.task_name],
            ['model_id', args.model_id],
            ['model', args.model],
            ['data', args.data],
            ['data_path', args.data_path],
            ['target', args.target],
            ['seq_len', args.seq_len],
            ['pred_len', args.pred_len],
            ['e_layers', args.e_layers],
            ['d_layers', args.d_layers],
            ['d_model', args.d_model],
            ['d_ff', args.d_ff],
            ['n_heads', args.n_heads],
            ['factor', args.factor],
            ['dropout', args.dropout],
            ['moving_avg', args.moving_avg],
            ['price_patch_scales', ','.join(map(str, args.price_patch_scales))],
            ['learning_rate', args.learning_rate],
            ['train_epochs', args.train_epochs],
            ['patience', args.patience],
            ['batch_size', args.batch_size],
            ['price_test_size', args.price_test_size],
            ['test_start_hour', args.test_start_hour],
            ['test_start_minute', args.test_start_minute],
            ['price_interval_minutes', args.price_interval_minutes],
            ['known_exo_dim', getattr(args, 'known_exo_dim', '')],
            ['known_hist_exo_features', getattr(args, 'known_hist_exo_features', '')],
            ['known_future_exo_features', getattr(args, 'known_future_exo_features', '')],
            [],
            ['loss_config', 'value'],
            ['loss_name', WeightedHuberLoss.__name__],
        ]
        rows.extend([['loss_{}'.format(name), value] for name, value in DEFAULT_LOSS_PARAMS.items()])
        return rows

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = self._move_batch_to_device(batch)
                outputs = self._unwrap_outputs(self.model(batch))
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch['price_future'].detach().cpu().numpy())

        if not preds:
            raise ValueError('test_loader produced no batches; check price_test_size, test_start_hour/minute, and drop_last')

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('scaled mse:{}, mae:{}'.format(mse, mae))

        pred_real = test_data.inverse_price(preds)
        true_real = test_data.inverse_price(trues)
        real_mae, real_mse, real_rmse, real_mape, real_mspe = metric(pred_real, true_real)
        print('price mse:{}, mae:{}'.format(real_mse, real_mae))

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + 'pred_scaled.npy', preds)
        np.save(folder_path + 'true_scaled.npy', trues)
        np.save(folder_path + 'pred.npy', pred_real)
        np.save(folder_path + 'true.npy', true_real)
        future_datetimes = self._test_future_datetimes(test_data, pred_real.shape[0])
        np.save(folder_path + 'datetime.npy', future_datetimes)
        self._save_prediction_table(folder_path, future_datetimes, pred_real, true_real)
        self._save_prediction_plot(folder_path, future_datetimes, pred_real, true_real)
        with open(folder_path + 'metrics.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['mae', real_mae])
            writer.writerow(['mse', real_mse])
            writer.writerow(['rmse', real_rmse])
            writer.writerow(['mape', real_mape])
            writer.writerow(['mspe', real_mspe])
            writer.writerows(self._metric_config_rows())
        with open(folder_path + 'metrics_scaled.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['mae', mae])
            writer.writerow(['mse', mse])
            writer.writerow(['rmse', rmse])
            writer.writerow(['mape', mape])
            writer.writerow(['mspe', mspe])
        return
