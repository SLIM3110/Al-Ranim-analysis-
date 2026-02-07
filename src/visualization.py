"""Shared visualization functions for Al-Ranim analysis."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd


def setup_style():
    """Set consistent matplotlib/seaborn style for all notebooks."""
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.dpi": 100,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
    })
    sns.set_palette("husl")


def format_aed(value, pos=None):
    """Format numbers as AED with K/M suffixes."""
    if abs(value) >= 1_000_000:
        return f"AED {value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"AED {value/1_000:.0f}K"
    return f"AED {value:.0f}"


def plot_price_distribution(df, column, hue=None, title="", ax=None, bins=30):
    """Histogram + KDE distribution plot."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    if hue:
        for name, group in df.groupby(hue):
            ax.hist(group[column].dropna(), bins=bins, alpha=0.5, label=str(name), density=True)
            try:
                group[column].dropna().plot.kde(ax=ax, label=f"{name} (KDE)", linewidth=2)
            except Exception:
                pass
        ax.legend()
    else:
        ax.hist(df[column].dropna(), bins=bins, alpha=0.6, color="#2E86AB", density=True, edgecolor="white")
        try:
            df[column].dropna().plot.kde(ax=ax, color="#E63946", linewidth=2)
        except Exception:
            pass

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
    ax.set_ylabel("Density")
    return ax


def plot_comparison_bars(df, x, y, hue=None, title="", ax=None, palette=None, fmt="AED"):
    """Grouped bar chart with value labels."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    if hue:
        chart = sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax, palette=palette, edgecolor="white")
    else:
        chart = sns.barplot(data=df, x=x, y=y, ax=ax, palette=palette, edgecolor="white")

    # Add value labels
    for container in ax.containers:
        labels = []
        for bar in container:
            val = bar.get_height()
            if fmt == "AED":
                if abs(val) >= 1_000_000:
                    labels.append(f"{val/1_000_000:.2f}M")
                elif abs(val) >= 1_000:
                    labels.append(f"{val/1_000:.0f}K")
                else:
                    labels.append(f"{val:.0f}")
            elif fmt == "pct":
                labels.append(f"{val:.1f}%")
            else:
                labels.append(f"{val:.1f}")
        ax.bar_label(container, labels=labels, padding=3, fontsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    return ax


def plot_correlation_heatmap(df, features, title="", figsize=(10, 8)):
    """Annotated heatmap of feature correlations."""
    fig, ax = plt.subplots(figsize=figsize)
    corr = df[features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, ax=ax, square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig, ax


def plot_feature_importance(importances, feature_names, model_name="", top_n=15, ax=None):
    """Horizontal bar chart of feature importances."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.4)))

    idx = np.argsort(importances)[-top_n:]
    ax.barh(
        range(len(idx)),
        importances[idx],
        color="#2E86AB",
        edgecolor="white",
    )
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance - {model_name}", fontsize=14, fontweight="bold")
    return ax


def plot_model_comparison(results_dict, metrics=None):
    """Side-by-side model performance comparison."""
    if metrics is None:
        metrics = ["R2", "RMSE", "MAE"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        models = list(results_dict.keys())
        values = [results_dict[m].get(metric, 0) for m in models]
        colors = sns.color_palette("husl", len(models))
        bars = ax.bar(range(len(models)), values, color=colors, edgecolor="white")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)

    plt.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_location_premium(df, location_col, value_col, title="", ax=None):
    """Box plot comparing values by location feature (e.g., park-facing vs not)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(data=df, x=location_col, y=value_col, ax=ax, palette=["#F4A261", "#2BA84A"])
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_aed))
    return ax


def create_summary_table(data, title=""):
    """Create a styled summary table figure."""
    fig, ax = plt.subplots(figsize=(12, max(2, len(data) * 0.5 + 1)))
    ax.axis("off")
    if isinstance(data, pd.DataFrame):
        table = ax.table(
            cellText=data.values,
            colLabels=data.columns,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        # Style header
        for j in range(len(data.columns)):
            table[0, j].set_facecolor("#1B3A5C")
            table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    return fig
