import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Autoformer_EncDec import series_decomp
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.StandardNorm import Normalize


class PriceMultiScalePatchEmbedding(nn.Module):
    def __init__(self, seq_len, d_model, patch_scales, dropout):
        super(PriceMultiScalePatchEmbedding, self).__init__()
        self.seq_len = seq_len
        self.patch_scales = patch_scales
        self.projections = nn.ModuleList([
            nn.Linear(scale, d_model) for scale in patch_scales
        ])
        self.position_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(1, self._num_patches(seq_len, scale), d_model) * 0.02)
            for scale in patch_scales
        ])
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _num_patches(seq_len, patch_len):
        return (seq_len + patch_len - 1) // patch_len

    @staticmethod
    def _pad_to_patch(x, patch_len):
        pad_len = (-x.size(1)) % patch_len
        if pad_len == 0:
            return x
        pad_value = x[:, -1:, :].repeat(1, pad_len, 1)
        return torch.cat([x, pad_value], dim=1)

    def forward(self, price_hist):
        # price_hist: [B, L, 1]
        outputs = []
        price_hist = price_hist.transpose(1, 2)  # [B, 1, L]

        for patch_len, projection, position in zip(
                self.patch_scales, self.projections, self.position_embeddings):
            x = self._pad_to_patch(price_hist.transpose(1, 2), patch_len)
            bsz, length, channels = x.shape
            x = x.transpose(1, 2).reshape(bsz * channels, length)
            x = x.unfold(dimension=-1, size=patch_len, step=patch_len)
            x = projection(x)
            x = x + position[:, :x.size(1), :]
            outputs.append(self.dropout(x))

        return outputs


class MultiScaleSeasonMixing(nn.Module):
    def __init__(self, d_model, num_scales):
        super(MultiScaleSeasonMixing, self).__init__()
        self.mixers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(num_scales - 1)
        ])

    def forward(self, season_list):
        # Fine to coarse: inject pooled fine-scale information into coarser tokens.
        outputs = [season_list[0]]
        running = season_list[0]
        for i, coarse in enumerate(season_list[1:]):
            pooled = F.adaptive_avg_pool1d(
                running.transpose(1, 2), coarse.size(1)
            ).transpose(1, 2)
            running = coarse + self.mixers[i](pooled)
            outputs.append(running)
        return outputs


class MultiScaleTrendMixing(nn.Module):
    def __init__(self, d_model, num_scales):
        super(MultiScaleTrendMixing, self).__init__()
        self.mixers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(num_scales - 1)
        ])

    def forward(self, trend_list):
        # Coarse to fine: inject broad trend information into finer tokens.
        outputs = [None] * len(trend_list)
        running = trend_list[-1]
        outputs[-1] = running
        for offset, fine in enumerate(reversed(trend_list[:-1])):
            mixer = self.mixers[offset]
            pooled = F.interpolate(
                running.transpose(1, 2), size=fine.size(1), mode="linear", align_corners=False
            ).transpose(1, 2)
            running = fine + mixer(pooled)
            outputs[len(trend_list) - 2 - offset] = running
        return outputs


class SeasonTrendMixer(nn.Module):
    def __init__(self, d_model, num_scales, moving_avg, dropout):
        super(SeasonTrendMixer, self).__init__()
        self.decomp = series_decomp(moving_avg)
        self.season_mixing = MultiScaleSeasonMixing(d_model, num_scales)
        self.trend_mixing = MultiScaleTrendMixing(d_model, num_scales)
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
            )
            for _ in range(num_scales)
        ])

    def forward(self, scale_tokens):
        season_list = []
        trend_list = []
        for tokens in scale_tokens:
            season, trend = self.decomp(tokens)
            season_list.append(season)
            trend_list.append(trend)

        season_list = self.season_mixing(season_list)
        trend_list = self.trend_mixing(trend_list)

        outputs = []
        for tokens, season, trend, layer in zip(scale_tokens, season_list, trend_list, self.output_layers):
            mixed = season + trend
            outputs.append(tokens + layer(mixed))
        return outputs


class ExoEncoderWithFuture(nn.Module):
    def __init__(self, known_dim, unknown_dim, time_dim, d_model, dropout):
        super(ExoEncoderWithFuture, self).__init__()
        self.known_dim = known_dim
        self.unknown_dim = unknown_dim
        self.time_dim = time_dim

        self.known_projection = nn.Linear(known_dim + time_dim, d_model)
        self.unknown_projection = nn.Linear(unknown_dim, d_model) if unknown_dim > 0 else None
        self.mask_projection = nn.Linear(unknown_dim, d_model) if unknown_dim > 0 else None
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model)) if unknown_dim > 0 else None
        self.fusion = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _check_dim(name, tensor, expected):
        if tensor.size(-1) != expected:
            raise ValueError(
                "{} last dimension is {}, expected {}".format(name, tensor.size(-1), expected)
            )

    def forward(self, batch):
        known_hist = batch["known_hist_exo"]
        known_future = batch["known_future_exo"]
        unknown_hist = batch["unknown_hist_exo"]
        unknown_future_mask = batch["unknown_future_mask"]
        time_hist = batch["time_hist"]
        time_future = batch["time_future"]

        self._check_dim("known_hist_exo", known_hist, self.known_dim)
        self._check_dim("known_future_exo", known_future, self.known_dim)
        self._check_dim("unknown_hist_exo", unknown_hist, self.unknown_dim)
        self._check_dim("unknown_future_mask", unknown_future_mask, self.unknown_dim)
        self._check_dim("time_hist", time_hist, self.time_dim)
        self._check_dim("time_future", time_future, self.time_dim)

        known_tokens = self.known_projection(
            torch.cat([
                torch.cat([known_hist, time_hist], dim=-1),
                torch.cat([known_future, time_future], dim=-1),
            ], dim=1)
        )

        if self.unknown_dim > 0:
            unknown_hist_tokens = self.unknown_projection(unknown_hist)
            unknown_future_tokens = self.mask_token + self.mask_projection(unknown_future_mask)
            unknown_tokens = torch.cat([unknown_hist_tokens, unknown_future_tokens], dim=1)
        else:
            unknown_tokens = torch.zeros_like(known_tokens)

        exo_tokens = known_tokens + unknown_tokens
        return self.dropout(exo_tokens + self.fusion(exo_tokens))


class PriceExoCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, factor, output_attention):
        super(PriceExoCrossAttention, self).__init__()
        self.attention = AttentionLayer(
            FullAttention(
                mask_flag=False,
                factor=factor,
                attention_dropout=dropout,
                output_attention=output_attention,
            ),
            d_model,
            n_heads,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, price_tokens, exo_tokens):
        attended, attn = self.attention(price_tokens, exo_tokens, exo_tokens, attn_mask=None)
        price_tokens = self.norm1(price_tokens + self.dropout(attended))
        price_tokens = self.norm2(price_tokens + self.dropout(self.ffn(price_tokens)))
        return price_tokens, attn


class MultiStepHead(nn.Module):
    def __init__(self, d_model, pred_len, dropout):
        super(MultiStepHead, self).__init__()
        self.pred_len = pred_len
        short_len, mid_len, long_len = self._segment_lengths(pred_len)
        self.segment_lengths = (short_len, mid_len, long_len)

        self.short_head = self._build_head(d_model, short_len, dropout)
        self.mid_head = self._build_head(d_model, mid_len, dropout)
        self.long_head = self._build_head(d_model, long_len, dropout)

    @staticmethod
    def _build_head(d_model, out_len, dropout):
        return nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, out_len),
        )

    @staticmethod
    def _segment_lengths(pred_len):
        if pred_len == 96:
            return 24, 24, 48
        short_len = max(1, pred_len // 4)
        mid_len = max(1, pred_len // 4)
        long_len = pred_len - short_len - mid_len
        if long_len < 1:
            long_len = 1
            mid_len = max(1, pred_len - short_len - long_len)
        return short_len, mid_len, long_len

    def forward(self, scale_tokens):
        fine_feature = scale_tokens[0].mean(dim=1)
        mid_feature = scale_tokens[min(1, len(scale_tokens) - 1)].mean(dim=1)
        coarse_feature = scale_tokens[-1].mean(dim=1)

        short_pred = self.short_head(fine_feature)
        mid_pred = self.mid_head(mid_feature)
        long_pred = self.long_head(coarse_feature)
        pred = torch.cat([short_pred, mid_pred, long_pred], dim=-1)
        return pred[:, :self.pred_len].unsqueeze(-1)


class FutureAttentionHead(nn.Module):
    def __init__(self, known_dim, time_dim, d_model, pred_len, n_heads, dropout, factor, output_attention):
        super(FutureAttentionHead, self).__init__()
        self.pred_len = pred_len
        self.future_projection = nn.Linear(known_dim + time_dim, d_model)
        self.register_buffer(
            "horizon_position_encoding",
            self._build_horizon_position_encoding(pred_len, d_model),
            persistent=False,
        )
        self.cross_attention = AttentionLayer(
            FullAttention(
                mask_flag=False,
                factor=factor,
                attention_dropout=dropout,
                output_attention=output_attention,
            ),
            d_model,
            n_heads,
        )
        self.self_attention = AttentionLayer(
            FullAttention(
                mask_flag=False,
                factor=factor,
                attention_dropout=dropout,
                output_attention=output_attention,
            ),
            d_model,
            n_heads,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.projection = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _build_horizon_position_encoding(pred_len, d_model):
        position = torch.arange(pred_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(pred_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
        return pe.unsqueeze(0)

    def forward(self, scale_tokens, exo_tokens, batch):
        future_inputs = torch.cat([batch["known_future_exo"], batch["time_future"]], dim=-1)
        future_tokens = self.future_projection(future_inputs)
        future_tokens = future_tokens + self.horizon_position_encoding[:, :future_tokens.size(1), :]

        memory_tokens = torch.cat(scale_tokens + [exo_tokens], dim=1)
        attended, cross_attn = self.cross_attention(future_tokens, memory_tokens, memory_tokens, attn_mask=None)
        future_tokens = self.norm1(future_tokens + self.dropout(attended))

        attended, self_attn = self.self_attention(future_tokens, future_tokens, future_tokens, attn_mask=None)
        future_tokens = self.norm2(future_tokens + self.dropout(attended))
        future_tokens = self.norm3(future_tokens + self.dropout(self.ffn(future_tokens)))

        pred = self.projection(future_tokens)
        return pred[:, :self.pred_len, :], [cross_attn, self_attn]


class Model(nn.Module):
    """
    TimeXer-style electricity price forecaster.

    Expected batch keys:
        price_hist: [B, seq_len, 1]
        known_hist_exo: [B, seq_len, known_exo_dim]
        known_future_exo: [B, pred_len, known_exo_dim]
        unknown_hist_exo: [B, seq_len, unknown_exo_dim]
        unknown_future_mask: [B, pred_len, unknown_exo_dim]
        time_hist: [B, seq_len, time_dim]
        time_future: [B, pred_len, time_dim]
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model

        self.known_exo_dim = getattr(configs, "known_exo_dim", 15)
        self.unknown_exo_dim = getattr(configs, "unknown_exo_dim", 7)
        self.time_dim = getattr(configs, "time_dim", 4)
        self.patch_scales = getattr(configs, "price_patch_scales", [4, 8, 16, 32])

        self.price_norm = Normalize(1, affine=True, non_norm=True if getattr(configs, "use_norm", 1) == 0 else False)
        self.price_embedding = PriceMultiScalePatchEmbedding(
            self.seq_len,
            self.d_model,
            self.patch_scales,
            configs.dropout,
        )
        self.season_trend_mixer = SeasonTrendMixer(
            self.d_model,
            len(self.patch_scales),
            configs.moving_avg,
            configs.dropout,
        )
        self.exo_encoder = ExoEncoderWithFuture(
            self.known_exo_dim,
            self.unknown_exo_dim,
            self.time_dim,
            self.d_model,
            configs.dropout,
        )
        self.cross_layers = nn.ModuleList([
            PriceExoCrossAttention(
                self.d_model,
                configs.n_heads,
                configs.dropout,
                configs.factor,
                configs.output_attention,
            )
            for _ in range(configs.e_layers)
        ])
        self.head = FutureAttentionHead(
            self.known_exo_dim,
            self.time_dim,
            self.d_model,
            self.pred_len,
            configs.n_heads,
            configs.dropout,
            configs.factor,
            configs.output_attention,
        )

    @staticmethod
    def _require_keys(batch, keys):
        missing = [key for key in keys if key not in batch]
        if missing:
            raise KeyError("TimeXerPrice batch is missing keys: {}".format(", ".join(missing)))

    def forecast(self, batch):
        required_keys = [
            "price_hist",
            "known_hist_exo",
            "known_future_exo",
            "unknown_hist_exo",
            "unknown_future_mask",
            "time_hist",
            "time_future",
        ]
        self._require_keys(batch, required_keys)

        price_hist = batch["price_hist"]
        if price_hist.size(-1) != 1:
            raise ValueError("price_hist last dimension must be 1, got {}".format(price_hist.size(-1)))

        price_hist = self.price_norm(price_hist, "norm")
        scale_tokens = self.price_embedding(price_hist)
        scale_tokens = self.season_trend_mixer(scale_tokens)

        exo_tokens = self.exo_encoder(batch)
        attentions = []
        for cross_layer in self.cross_layers:
            next_tokens = []
            for tokens in scale_tokens:
                mixed_tokens, attn = cross_layer(tokens, exo_tokens)
                next_tokens.append(mixed_tokens)
                attentions.append(attn)
            scale_tokens = self.season_trend_mixer(next_tokens)

        pred, head_attentions = self.head(scale_tokens, exo_tokens, batch)
        pred = self.price_norm(pred, "denorm")
        if getattr(self.configs, "output_attention", False):
            attentions.extend(head_attentions)
            return pred, attentions
        return pred

    def forward(self, batch):
        if self.task_name in ["price_forecast", "long_term_forecast", "short_term_forecast"]:
            return self.forecast(batch)
        raise ValueError("TimeXerPrice only supports forecasting tasks")
