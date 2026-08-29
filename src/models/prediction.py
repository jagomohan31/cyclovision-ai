"""
Model C — Prediction.

Forecasts how the cyclone evolves over the next SEQUENCE_LENGTH_OUT steps
(24-72 hours at typical 6-12h track-update cadence), given the past
SEQUENCE_LENGTH_IN frames/observations.

Two complementary heads share one ConvLSTM encoder over the cropped image
sequence:
  1. `forecast_frames`   — future satellite frames (spatial track/shape
                            evolution), useful for visualising the storm's
                            likely path as a "track cone" on the dashboard.
  2. `forecast_track`    — a compact (lat_offset, lon_offset, wind_kmh) at
                            each future step, which is what you actually
                            score against IBTrACS ground truth (track error
                            in km, intensity error in km/h).

A ConvLSTM cell is a standard LSTM but with its gate computations done as
2D convolutions instead of fully-connected layers, so it preserves spatial
structure across time — exactly what "predict how this cloud pattern moves
and intensifies" needs.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        # One conv produces all 4 gates (input, forget, output, cell) at once
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.hidden_channels = hidden_channels

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        shape = (batch_size, self.hidden_channels, height, width)
        return (torch.zeros(shape, device=device), torch.zeros(shape, device=device))


class CyclonePredictor(nn.Module):
    """
    Args:
        in_channels: satellite channels per frame (e.g. 1 for IR-only).
        hidden_channels: ConvLSTM hidden state width.
        seq_len_out: how many future steps to forecast.
        track_features: size of the per-step track vector, default 3
            for (lat_offset, lon_offset, wind_speed_kmh).
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        seq_len_out: int = 4,
        track_features: int = 3,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.seq_len_out = seq_len_out

        self.encoder_cell = ConvLSTMCell(in_channels, hidden_channels)
        self.decoder_cell = ConvLSTMCell(in_channels, hidden_channels)

        # Decoder hidden state -> next frame
        self.frame_head = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)

        # Decoder hidden state -> compact track/intensity vector for this step
        self.track_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, track_features),
        )

    def forward(self, x_seq: torch.Tensor):
        """
        x_seq: (B, T_in, C, H, W) — past satellite frames, time-ordered.
        Returns:
            forecast_frames: (B, seq_len_out, C, H, W)
            forecast_track:  (B, seq_len_out, track_features)
        """
        b, t_in, c, h, w = x_seq.shape
        device = x_seq.device

        h_t, c_t = self.encoder_cell.init_hidden(b, h, w, device)
        for t in range(t_in):
            h_t, c_t = self.encoder_cell(x_seq[:, t], h_t, c_t)

        # Decode seq_len_out future steps autoregressively, feeding each
        # predicted frame back in as the next step's input.
        last_frame = x_seq[:, -1]
        frames_out, track_out = [], []
        for _ in range(self.seq_len_out):
            h_t, c_t = self.decoder_cell(last_frame, h_t, c_t)
            next_frame = self.frame_head(h_t)
            next_track = self.track_head(h_t)
            frames_out.append(next_frame)
            track_out.append(next_track)
            last_frame = next_frame

        forecast_frames = torch.stack(frames_out, dim=1)   # (B, seq_len_out, C, H, W)
        forecast_track = torch.stack(track_out, dim=1)      # (B, seq_len_out, track_features)
        return forecast_frames, forecast_track


if __name__ == "__main__":
    model = CyclonePredictor(in_channels=1, hidden_channels=16, seq_len_out=4)
    dummy_seq = torch.randn(2, 8, 1, 64, 64)  # batch=2, 8 past frames, 64x64
    frames, track = model(dummy_seq)

    print("Input sequence shape:", dummy_seq.shape)
    print("Forecast frames shape:", frames.shape)
    print("Forecast track shape (lat_off, lon_off, wind_kmh) per step:", track.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
