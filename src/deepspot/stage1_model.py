from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from deepspot.stage1_data import Stage1Arrays


class Stage1WindowDataset(Dataset):
    def __init__(
        self,
        arrays: Stage1Arrays,
        indices: np.ndarray | None = None,
        event_mean: np.ndarray | None = None,
        event_std: np.ndarray | None = None,
    ):
        self.arrays = arrays
        self.indices = np.arange(len(arrays.raw_signal), dtype=np.int64) if indices is None else indices.astype(np.int64)
        self.event_mean = event_mean
        self.event_std = event_std

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        raw = self.arrays.raw_signal[idx].astype(np.float32)
        events = self.arrays.event_features[idx].astype(np.float32)
        mask = self.arrays.event_mask[idx].astype(np.float32)

        if self.event_mean is not None and self.event_std is not None:
            events = (events - self.event_mean.reshape(1, -1)) / self.event_std.reshape(1, -1)
            events = events * mask[:, None]

        return {
            "raw": torch.from_numpy(raw).unsqueeze(0),
            "events": torch.from_numpy(events),
            "mask": torch.from_numpy(mask),
            "index": torch.tensor(idx, dtype=torch.long),
        }


class RawEventAutoencoder(nn.Module):
    def __init__(
        self,
        window_len: int,
        max_events: int,
        event_dim: int,
        latent_dim: int = 48,
        event_hidden: int = 64,
    ):
        super().__init__()
        if window_len % 8 != 0:
            raise ValueError(f"window_len must be divisible by 8, got {window_len}")

        self.window_len = int(window_len)
        self.max_events = int(max_events)
        self.event_dim = int(event_dim)
        self.latent_dim = int(latent_dim)
        self.reduced_len = window_len // 8

        self.raw_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.raw_flat_dim = 64 * self.reduced_len
        self.raw_proj = nn.Sequential(
            nn.Linear(self.raw_flat_dim, 96),
            nn.GELU(),
        )

        self.event_token = nn.Sequential(
            nn.Linear(event_dim, event_hidden),
            nn.GELU(),
            nn.Linear(event_hidden, event_hidden),
            nn.GELU(),
        )
        self.event_proj = nn.Sequential(
            nn.Linear(event_hidden, 64),
            nn.GELU(),
        )

        self.to_latent = nn.Linear(96 + 64, latent_dim)

        self.raw_from_latent = nn.Linear(latent_dim, self.raw_flat_dim)
        self.raw_decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
        )

        self.event_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, max_events * event_dim),
        )

    def encode(self, raw: torch.Tensor, events: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raw_h = self.raw_encoder(raw).flatten(start_dim=1)
        raw_h = self.raw_proj(raw_h)

        token_h = self.event_token(events)
        mask_f = mask.unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        event_h = (token_h * mask_f).sum(dim=1) / denom
        event_h = self.event_proj(event_h)

        return self.to_latent(torch.cat([raw_h, event_h], dim=1))

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_h = self.raw_from_latent(z).view(z.size(0), 64, self.reduced_len)
        raw_hat = self.raw_decoder(raw_h)
        event_hat = self.event_decoder(z).view(z.size(0), self.max_events, self.event_dim)
        return raw_hat, event_hat

    def forward(self, raw: torch.Tensor, events: torch.Tensor, mask: torch.Tensor):
        z = self.encode(raw, events, mask)
        raw_hat, event_hat = self.decode(z)
        return raw_hat, event_hat, z


def reconstruction_losses(
    raw: torch.Tensor,
    raw_hat: torch.Tensor,
    events: torch.Tensor,
    event_hat: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_mse = ((raw - raw_hat) ** 2).mean(dim=(1, 2))
    event_sq = ((events - event_hat) ** 2) * mask.unsqueeze(-1)
    event_den = (mask.sum(dim=1) * events.size(-1)).clamp_min(1.0)
    event_mse = event_sq.sum(dim=(1, 2)) / event_den
    return raw_mse, event_mse
