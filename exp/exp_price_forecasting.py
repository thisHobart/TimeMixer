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

warnings.filterwarnings('ignore')


class WeightedHuberLoss(nn.Module):
    def __init__(self, spike_threshold=0.9, spike_weight=2.5,
                 short_weight=1.1, mid_weight=1.0, long_weight=0.9,
                 change_loss_weight=0.2, delta=0.25):
        super(WeightedHuberLoss, self).__init__()
        self.spike_threshold = spike_threshold
        self.spike_weight = spike_weight
        self.short_weight = short_weight
        self.mid_weight = mid_weight
        self.long_weight = long_weight
        self.change_loss_weight = change_loss_weight
        self.delta = delta

    def _huber(self, pred, true):
        error = pred - true
        abs_error = torch.abs(error)
        quadratic = torch.minimum(abs_error, torch.tensor(self.delta, device=pred.device, dtype=pred.dtype))
        linear = abs_error - quadratic
        return 0.5 * quadratic * quadratic + self.delta * linear

    def _step_weight(self, pred_len, device, dtype):
        short_len = min(24, pred_len)
        mid_len = min(24, max(0, pred_len - short_len))
        long_len = max(0, pred_len - short_len - mid_len)
        weights = []
        if short_len:
            weights.append(torch.full((short_len,), self.short_weight, device=device, dtype=dtype))
        if mid_len:
            weights.append(torch.full((mid_len,), self.mid_weight, device=device, dtype=dtype))
        if long_len:
            weights.append(torch.full((long_len,), self.long_weight, device=device, dtype=dtype))
        return torch.cat(weights).view(1, pred_len, 1)

    def _spike_weight(self, true):
        prev = torch.cat([true[:, :1, :], true[:, :-1, :]], dim=1)
        change = torch.abs(true - prev)
        return torch.where(
            change >= self.spike_threshold,
            torch.full_like(true, self.spike_weight),
            torch.ones_like(true),
        )

    def forward(self, pred, true):
        pred_len = pred.size(1)
        weights = self._step_weight(pred_len, pred.device, pred.dtype) * self._spike_weight(true)
        level_loss = (self._huber(pred, true) * weights).mean()

        pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
        true_diff = true[:, 1:, :] - true[:, :-1, :]
        change_loss = self._huber(pred_diff, true_diff).mean()
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
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return WeightedHuberLoss(
            spike_threshold=40.0,
            spike_weight=4.5,
            short_weight=1.5,
            mid_weight=1.0,
            long_weight=0.8,
            change_loss_weight=0.3,
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
        criterion = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
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

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} EarlyStop Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, train_loss, test_loss))
            loss_history.append([epoch + 1, train_loss, train_loss, test_loss])
            early_stopping(train_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

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
        with open(folder_path + 'metrics.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['mae', real_mae])
            writer.writerow(['mse', real_mse])
            writer.writerow(['rmse', real_rmse])
            writer.writerow(['mape', real_mape])
            writer.writerow(['mspe', real_mspe])
        with open(folder_path + 'metrics_scaled.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['mae', mae])
            writer.writerow(['mse', mse])
            writer.writerow(['rmse', rmse])
            writer.writerow(['mape', mape])
            writer.writerow(['mspe', mspe])
        return
