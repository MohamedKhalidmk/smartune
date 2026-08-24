"""
training/forecasting.py

Arm A (Domhan-style parametric curve extrapolation, Domhan et al. 2015)
is the sole forecasting method in the pipeline going forward.

This replaces an earlier Arm B (LC-PFN, Adriaensen et al. 2023)
implementation. A 10-curve real-fine-tuning-run comparison (see
results/forecasting_experiment_summary*.json) found Arm A more
accurate on average and in head-to-head win-count at every tested
epoch cutoff except one where the two were statistically
indistinguishable (overlapping error bars on a 30-sample slice) —
and Arm A's advantage grows with more observed epochs, since LC-PFN's
edge (if any) only shows up in the very-short-curve regime. Arm A
also drops the lcpfn dependency entirely (no PyPI version-pin bypass,
no runtime monkey-patches for modern PyTorch — see the removed
comments in requirements.txt).

This module produces:
- a point forecast + uncertainty band from an ensemble of three
  parametric curve fits (pow3 / exp3 / log_power), weighted by fit
  quality
- a DA-LCE-style difficulty proxy from the curve's early dynamics
- a plot of the observed curve + forecast + uncertainty

None of this makes a stop/continue decision. That is handled by
decision_engine.py.
"""

import matplotlib
import numpy as np
from scipy.optimize import curve_fit


# ============================================================
# ARM A — parametric curve extrapolation
# ============================================================

def _pow3(x, a, b, c):
    return a + b * np.power(x, -c)


def _exp3(x, a, b, c):
    return a + b * np.exp(-c * x)


def _log_power(x, a, b, c):
    return a + (1 - a) / (
        1 + np.power(np.maximum(x, 1e-6) / max(b, 1e-6), c)
    )


_CURVE_FAMILIES = {
    "pow3": (_pow3, [1.0, 1.0, 0.5]),
    "exp3": (_exp3, [1.0, 1.0, 0.1]),
    "log_power": (_log_power, [0.5, 5.0, 1.0]),
}

# z-score for a 90% interval (5th-95th percentile), used to turn the
# ensemble's fit-residual std into lower/upper bounds so the output
# shape matches what decision_engine.py expects (median/lower_5/
# upper_95), the same shape the earlier LC-PFN implementation produced.
_Z_90 = 1.645


def arm_a_forecast_with_uncertainty(
    val_losses: list[float],
    horizons: list[int],
) -> dict:
    """
    Forecast at each horizon by fitting three parametric curve families
    (pow3, exp3, log_power) to the observed validation-loss curve,
    combining them into a weighted ensemble (weighted by fit quality,
    i.e. lower sum-of-squared-residuals gets more weight), and
    extrapolating forward.

    Returns:
        {
            "median": [...],
            "lower_5": [...],
            "upper_95": [...],
        }
    """
    x_observed = np.arange(1, len(val_losses) + 1, dtype=float)
    y_observed = np.array(val_losses, dtype=float)

    max_horizon = max(horizons)
    x_future = np.arange(
        len(val_losses) + 1,
        len(val_losses) + max_horizon + 1,
        dtype=float,
    )

    fitted = []

    for name, (func, p0) in _CURVE_FAMILIES.items():
        try:
            params, _ = curve_fit(
                func, x_observed, y_observed, p0=p0, maxfev=5000
            )

            fitted_y = func(x_observed, *params)
            residuals = fitted_y - y_observed
            sse = float(np.sum(residuals ** 2))
            residual_std = float(np.std(residuals)) + 1e-6

            future_y = func(x_future, *params)

            if np.any(np.isnan(future_y)) or np.any(np.isinf(future_y)):
                continue

            fitted.append((sse, residual_std, future_y))

        except (RuntimeError, ValueError, TypeError):
            continue

    if not fitted:
        # No curve family converged — fall back to a flat forecast at
        # the last observed value, with a wide uncertainty band since
        # this is a low-confidence fallback, not a real fit.
        flat = [val_losses[-1]] * len(horizons)
        return {
            "median": flat,
            "lower_5": [v - _Z_90 for v in flat],
            "upper_95": [v + _Z_90 for v in flat],
        }

    sses = np.array([f[0] for f in fitted])
    weights = np.exp(-sses / (np.std(sses) + 1e-8))
    weights = weights / weights.sum()

    combined_future = sum(w * f[2] for w, f in zip(weights, fitted))
    combined_std = sum(w * f[1] for w, f in zip(weights, fitted))

    median = [combined_future[h - 1] for h in horizons]

    return {
        "median": median,
        "lower_5": [v - _Z_90 * combined_std for v in median],
        "upper_95": [v + _Z_90 * combined_std for v in median],
    }


def compute_difficulty_proxy(
    early_losses: list[float],
) -> dict:
    """
    Compute a DA-LCE-style difficulty signal from the curve's own
    early dynamics.

    Returns:
        {
            "prog": float,
            "nonlin": float,
            "vol": float,
        }
    """
    y = np.array(early_losses, dtype=float)
    t_count = len(y)

    if t_count < 2:
        return {
            "prog": 0.0,
            "nonlin": 0.0,
            "vol": 0.0,
        }

    progress = float(
        (y[-1] - y[0]) / t_count
    )

    t = np.arange(t_count)

    linear_fit = (
        y[0]
        + (y[-1] - y[0]) / (t_count - 1) * t
    )

    nonlinearity = float(
        np.mean((y - linear_fit) ** 2)
    )

    diffs = np.diff(y)
    volatility = (
        float(np.std(diffs))
        if len(diffs) > 0
        else 0.0
    )

    return {
        "prog": progress,
        "nonlin": nonlinearity,
        "vol": volatility,
    }


def noise_floor(values: list[float]) -> float:
    """
    Estimate the noise floor using detrended residuals.

    Returns:
        Standard deviation of residuals after fitting a linear trend.
    """
    n = len(values)

    if n < 3:
        return 0.0

    x = np.arange(n)

    slope, intercept = np.polyfit(
        x,
        values,
        1,
    )

    residuals = np.array(values) - (
        slope * x + intercept
    )

    return float(np.std(residuals))


def forecast_n_epochs_ahead(
    val_losses: list[float],
    n_epochs_ahead: int = 3,
    save_plot_path: str | None = "forecast_plot.png",
) -> dict:
    """
    Forecast exactly `n_epochs_ahead` epochs into the future.

    Returns:
        {
            "forecast": {...},
            "horizons": [...],
            "plot_path": str | None,
        }
    """
    if n_epochs_ahead < 1:
        raise ValueError(
            "n_epochs_ahead must be at least 1."
        )

    horizons = list(
        range(1, n_epochs_ahead + 1)
    )

    forecast = arm_a_forecast_with_uncertainty(
        val_losses,
        horizons,
    )

    plot_path = None

    if save_plot_path:
        plot_path = plot_forecast(
            val_losses,
            forecast,
            horizons,
            save_plot_path,
        )

    return {
        "forecast": forecast,
        "horizons": horizons,
        "plot_path": plot_path,
    }


def plot_forecast(
    val_losses: list[float],
    forecast: dict,
    horizons: list[int],
    save_path: str = "forecast_plot.png",
) -> str:
    """
    Plot the observed validation curve, forecast median, and
    5-95% uncertainty band.

    Returns:
        Path to the saved plot.
    """
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    observed_x = list(
        range(1, len(val_losses) + 1)
    )

    forecast_x = [
        len(val_losses) + h
        for h in horizons
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        observed_x,
        val_losses,
        "k-o",
        label="observed",
        linewidth=2,
    )

    # Prepend the last observed point so the dashed forecast line visually
    # connects to the solid observed line instead of leaving a gap.
    bridge_x = [observed_x[-1]] + forecast_x
    bridge_median = [val_losses[-1]] + list(forecast["median"])

    ax.plot(
        bridge_x,
        bridge_median,
        "b--o",
        label="forecast (median)",
        alpha=0.8,
    )

    ax.fill_between(
        forecast_x,
        forecast["lower_5"],
        forecast["upper_95"],
        color="blue",
        alpha=0.15,
        label="5-95% interval",
    )

    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss")
    ax.set_title(
        "Validation loss: observed + forecast"
    )
    ax.legend()

    fig.savefig(
        save_path,
        dpi=120,
        bbox_inches="tight",
    )

    plt.close(fig)

    return save_path