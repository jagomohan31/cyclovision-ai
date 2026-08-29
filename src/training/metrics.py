"""
Evaluation metrics for all three models. These are deliberately the same
metrics operational meteorology uses (not just generic ML accuracy), which
is what makes the evaluation slide of a pitch credible to judges:

  - Detection:      IoU (Intersection over Union) between predicted and
                     true cyclone cloud-system mask.
  - Classification:  per-class F1 (accuracy alone hides failure on rare,
                     high-impact categories like Super Cyclonic Storm).
  - Prediction:      track error in kilometres (great-circle distance
                     between predicted and true storm centre) and
                     intensity error in km/h — exactly what IMD/NHC report.
"""

from __future__ import annotations
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Detection metric
# ---------------------------------------------------------------------------
def iou_score(pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """
    Intersection-over-Union between a predicted probability mask and a
    binary ground-truth mask. Both shaped (B, 1, H, W) or (B, H, W).
    """
    pred_bin = (pred_mask > threshold).float()
    true_bin = (true_mask > threshold).float()

    intersection = (pred_bin * true_bin).sum(dim=list(range(1, pred_bin.dim())))
    union = ((pred_bin + true_bin) > 0).float().sum(dim=list(range(1, pred_bin.dim())))

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------
def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[int, float]:
    """
    F1 score per class. Reported per-class (not just overall accuracy)
    because a model can score >80% accuracy while never once correctly
    flagging a Super Cyclonic Storm, simply because that class is rare —
    exactly the failure mode this metric is meant to catch.
    """
    scores = {}
    for c in range(num_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        scores[c] = f1
    return scores


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Simple confusion_matrix[true][pred] counts, no sklearn dependency needed."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


# ---------------------------------------------------------------------------
# Prediction metrics
# ---------------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """
    Great-circle distance in km between two (lat, lon) points/arrays —
    this is the standard "track error" metric IMD/NHC bulletins report.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return EARTH_RADIUS_KM * c


def track_and_intensity_error(
    pred_lat: np.ndarray, pred_lon: np.ndarray, pred_wind_kmh: np.ndarray,
    true_lat: np.ndarray, true_lon: np.ndarray, true_wind_kmh: np.ndarray,
) -> dict[str, float]:
    """
    Mean track error (km) and mean absolute intensity error (km/h) across
    a batch of forecasted steps. Report this broken down by lead time
    (6h, 12h, ..., 72h) in your actual results table — error should grow
    with lead time, and showing that honestly is more credible than
    hiding it.
    """
    track_err = haversine_km(pred_lat, pred_lon, true_lat, true_lon)
    intensity_err = np.abs(pred_wind_kmh - true_wind_kmh)
    return {
        "mean_track_error_km": float(np.mean(track_err)),
        "median_track_error_km": float(np.median(track_err)),
        "mean_intensity_error_kmh": float(np.mean(intensity_err)),
    }


if __name__ == "__main__":
    # Self-test with synthetic data
    pred_mask = torch.rand(4, 1, 32, 32)
    true_mask = (torch.rand(4, 1, 32, 32) > 0.5).float()
    print("IoU (random vs random, sanity check):", round(iou_score(pred_mask, true_mask), 3))

    y_true = np.array([0, 1, 2, 2, 3, 7, 7, 1])
    y_pred = np.array([0, 1, 2, 1, 3, 7, 5, 1])
    f1s = per_class_f1(y_true, y_pred, num_classes=8)
    print("Per-class F1:", {k: round(v, 2) for k, v in f1s.items() if v > 0 or k in y_true})

    err = track_and_intensity_error(
        pred_lat=np.array([15.1, 18.4]), pred_lon=np.array([84.2, 87.9]), pred_wind_kmh=np.array([95.0, 130.0]),
        true_lat=np.array([15.0, 18.0]), true_lon=np.array([84.0, 88.0]), true_wind_kmh=np.array([90.0, 140.0]),
    )
    print("Track/intensity error:", {k: round(v, 2) for k, v in err.items()})
