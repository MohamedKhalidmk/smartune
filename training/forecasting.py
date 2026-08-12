"""
training/forecasting.py

Arm B (LC-PFN, Adriaensen et al. 2023) is the sole forecasting method
in the pipeline going forward.

This module produces:
- a point forecast + uncertainty band from LC-PFN's posterior
- a DA-LCE-style difficulty proxy from the curve's early dynamics
- a plot of the observed curve + forecast + uncertainty

None of this makes a stop/continue decision. That is handled by
decision_engine.py.
"""

import matplotlib
import numpy as np
import torch
import lcpfn

from lcpfn import utils as lcpfn_utils


def arm_b_forecast_with_uncertainty(
    val_losses: list[float],
    horizons: list[int],
) -> dict:
    """
    Forecast at each horizon with a real uncertainty band from LC-PFN's
    own posterior predictive distribution (5th/50th/95th percentile).

    Returns:
        {
            "median": [...],
            "lower_5": [...],
            "upper_95": [...],
        }
    """
    model = lcpfn.LCPFN()

    x_train = torch.arange(
        1,
        len(val_losses) + 1,
        dtype=torch.float32,
    )

    y_train = torch.tensor(
        val_losses,
        dtype=torch.float32,
    )

    x_test = torch.tensor(
        [len(val_losses) + h for h in horizons],
        dtype=torch.float32,
    )

    normalizer = lcpfn_utils.pfn_normalize(
        lb=torch.tensor(0.0),
        ub=torch.tensor(float("inf")),
        soft_lb=0.0,
        soft_ub=torch.tensor(val_losses[0]),
        minimize=True,
    )

    y_train_norm = normalizer[0](y_train)

    with torch.no_grad():
        logits = model(
            x_train=x_train,
            y_train=y_train_norm,
            x_test=x_test,
        )

        results = {}

        for name, q in [
            ("lower_5", 0.05),
            ("median", 0.5),
            ("upper_95", 0.95),
        ]:
            value = model.model.criterion.icdf(logits, q)

            # Defensive shape fix for LC-PFN output.
            if value.dim() == 1:
                value = value.unsqueeze(1)

            value = normalizer[1](value)

            result = value.squeeze().tolist()
            results[name] = (
                result if isinstance(result, list) else [result]
            )

    return results


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

    forecast = arm_b_forecast_with_uncertainty(
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

    ax.plot(
        forecast_x,
        forecast["median"],
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
