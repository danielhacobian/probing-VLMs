"""Regenerate the three summary figures from committed numerical results."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
RATES = (("1e-5", "#2476ad"), ("1e-6", "#dc762e"))
PDF_METADATA = {"CreationDate": None, "ModDate": None}


def save(fig, name):
    fig.savefig(FIGURES / name, metadata=PDF_METADATA)
    plt.close(fig)


def read_deltas():
    return {
        path.parent.name: pd.read_csv(path)
        for path in HERE.glob("*/headline_straightening_deltas.csv")
    }


def predictor_deltas(tables, setting, variables):
    data = tables[setting]
    return data[data.family == "predictor"].set_index("variable").loc[variables]


def plot_deltas(tables):
    fig, axes = plt.subplots(1, 3, figsize=(7, 3), layout="constrained")
    targets = {
        "umaze": ["position", "velocity", "acceleration"],
        "wall": ["position", "velocity", "acceleration"],
        "pusht": ["block_position", "block_velocity", "block_acceleration",
                  "block_orientation", "block_angular_velocity"],
    }
    titles = ["UMaze agent", "Wall agent", "PushT block"]
    for ax, (environment, variables), title in zip(axes, targets.items(), titles):
        for (rate, color), offset in zip(RATES, [-0.13, 0.13]):
            data = predictor_deltas(tables, f"{environment}_{rate}", variables)
            delta = data.on_minus_off_r2.to_numpy()
            error = np.stack([
                delta - data.on_minus_off_r2_ci_low.to_numpy(),
                data.on_minus_off_r2_ci_high.to_numpy() - delta,
            ])
            ax.errorbar(delta, np.arange(len(data)) + offset, xerr=error,
                        fmt="o", capsize=2, color=color, markersize=4,
                        label=rf"$10^{{{rate[-2:]}}}$")
        labels = [v.removeprefix("block_").replace("_", " ")
                  .replace("acceleration", "accel.")
                  .replace("angular velocity", "angular vel.") for v in variables]
        ax.set_yticks(np.arange(len(variables)), labels)
        ax.invert_yaxis()
        ax.axvline(0, color="0.6", linewidth=0.8)
        ax.set(title=title, xlabel=r"$\Delta R^2$ (ON $-$ OFF)")
        ax.grid(axis="x", alpha=0.15)
    fig.legend(*axes[0].get_legend_handles_labels(),
               title="Matched projection learning rate",
               loc="outside upper center", ncol=2)
    save(fig, "locked_test_deltas.pdf")


def plot_pusht(tables):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
    variables = ["block_position", "block_velocity", "block_acceleration",
                 "block_orientation", "block_angular_velocity"]
    labels = ["Position", "Velocity", "Accel.", "Orientation", "Angular\nvelocity"]
    for ax, (rate, _) in zip(axes, RATES):
        data = predictor_deltas(tables, f"pusht_{rate}", variables)
        x = np.arange(len(variables))
        ax.bar(x - 0.18, data.off_r2, width=0.35, color="#666666", label="OFF")
        ax.bar(x + 0.18, data.on_r2, width=0.35, color="#d27a35", label="ON")
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set(ylim=(0, 0.85), ylabel="Locked-test $R^2$",
               title=rf"PushT, matched rate $10^{{{rate[-2:]}}}$")
        ax.legend(frameon=False, ncol=2)
    save(fig, "pusht_tradeoff.pdf")


def plot_transfer():
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
    panels = [("umaze_1e-5", "velocity", r"UMaze agent velocity, $10^{-5}$"),
              ("pusht_1e-6", "block_velocity", r"PushT block velocity, $10^{-6}$")]
    for ax, (setting, variable, title) in zip(axes, panels):
        data = pd.read_csv(HERE / setting / "matched_layerwise_cartesian_metrics.csv")
        data = data[(data.family == "predictor") & (data.variable == variable)]
        for condition, color in [("off", "#377eb8"), ("on", "#de7c2a")]:
            for split, style, label in [
                ("episode_validation", "-", "trajectory validation"),
                ("spatial_validation", "--", "spatial validation"),
            ]:
                rows = data[(data.condition == condition) & (data.split == split)]
                rows = rows.sort_values("layer")
                ax.plot(rows.layer, rows.r2, color=color, linestyle=style,
                        marker="o", markersize=3, label=f"{condition.upper()}, {label}")
        ax.axhline(0, color="0.5", linewidth=0.7)
        ax.set(title=title, xlabel="Predictor block", ylabel="Validation $R^2$")
        ax.grid(alpha=0.15)
    fig.legend(*axes[0].get_legend_handles_labels(),
               loc="outside upper center", ncol=2, frameon=False)
    save(fig, "depth_and_transfer.pdf")


def main():
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "legend.fontsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    tables = read_deltas()
    plot_deltas(tables)
    plot_pusht(tables)
    plot_transfer()


if __name__ == "__main__":
    main()
