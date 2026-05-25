# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================================
# CONFIGURATION
# ============================================================
SUMMARY_CSV = "/srv/lustre01/project/assiwat-tbwgr0oduuk/users/mohamed.benaly/outputsimulation/csvetfigures/projection_summary_from_detailcsv_merged_4db_incrementalIWUE.csv"
OUT_DIR = "/srv/lustre01/project/assiwat-tbwgr0oduuk/users/mohamed.benaly/outputsimulation/csvetfigures"
os.makedirs(OUT_DIR, exist_ok=True)

SCENARIOS_TO_KEEP = [
    "SSP245 - 2050",
    "SSP245 - 2100",
    "SSP585 - 2050",
    "SSP585 - 2100",
]

YIELD_SUFFICIENCY_THRESHOLD = 0.95
IWUE_COMPARABLE_TOL_REL = 0.01

DISPLAY_LEVELS_IRR = [100, 85, 80, 70, 50, 30, 0]
DISPLAY_LEVELS_N = [100, 85, 80, 70, 50, 30, 0]

# 100% N = real N dose
N_FULL_RATE_KG_HA = 210.0

OUT_BIOMASS_PNG = os.path.join(OUT_DIR, "projection_biomass_all_scenarios.png")
OUT_IRRIGATION_PNG = os.path.join(OUT_DIR, "projection_irrigation_all_scenarios.png")
OUT_IWUE_PNG = os.path.join(OUT_DIR, "projection_IWUE_all_scenarios.png")
OUT_NRES_PNG = os.path.join(OUT_DIR, "projection_Nres_all_scenarios.png")

OUT_BIOMASS_VALUES_PNG = os.path.join(OUT_DIR, "projection_biomass_all_scenarios_values.png")
OUT_IRRIGATION_VALUES_PNG = os.path.join(OUT_DIR, "projection_irrigation_all_scenarios_values.png")
OUT_IWUE_VALUES_PNG = os.path.join(OUT_DIR, "projection_IWUE_all_scenarios_values.png")
OUT_NRES_VALUES_PNG = os.path.join(OUT_DIR, "projection_Nres_all_scenarios_values.png")

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 10,
    "figure.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

TEXT_COLOR = "black"
NF_TEXT = "(NF)"


# ============================================================
# HELPERS
# ============================================================
def compute_optimum(stats_scenario: pd.DataFrame) -> pd.Series:
    feasible = stats_scenario[stats_scenario["Feasible"]].copy()
    if feasible.empty:
        raise ValueError("No feasible treatment for this scenario.")

    feasible = feasible.dropna(subset=["IWUE_mean_kgm3"]).copy()
    if feasible.empty:
        raise ValueError("No feasible treatment with valid IWUE for this scenario.")

    iwue_max = feasible["IWUE_mean_kgm3"].max()
    iwue_cutoff = iwue_max * (1 - IWUE_COMPARABLE_TOL_REL)

    comparable = feasible[feasible["IWUE_mean_kgm3"] >= iwue_cutoff].copy()
    comparable = comparable.sort_values(
        by=["Nres_mean", "IWUE_mean_kgm3", "N_pct", "Irr_pct"],
        ascending=[True, False, True, True],
    )
    return comparable.iloc[0]


def matrix_from_stats(stats_df: pd.DataFrame, value_col: str, irr_order: list[float], n_order: list[float]) -> np.ndarray:
    mat = (
        stats_df.pivot_table(index="Irr_pct", columns="N_pct", values=value_col, aggfunc="mean")
        .reindex(index=irr_order, columns=n_order)
    )
    return mat.values


def matrix_bool_from_stats(stats_df: pd.DataFrame, value_col: str, irr_order: list[float], n_order: list[float]) -> np.ndarray:
    mat = (
        stats_df.pivot_table(index="Irr_pct", columns="N_pct", values=value_col, aggfunc="first")
        .reindex(index=irr_order, columns=n_order)
    )
    return mat.fillna(False).astype(bool).values


def add_grid(ax, nrows: int, ncols: int) -> None:
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)


def select_levels_with_optimum(all_levels: list[float], preferred_levels: list[int], optimum_levels: list[float]) -> list[float]:
    all_levels_set = {float(v) for v in all_levels}
    selected = [float(v) for v in preferred_levels if float(v) in all_levels_set]

    for v in optimum_levels:
        selected.append(float(v))

    if all_levels:
        selected.append(float(min(all_levels)))
        selected.append(float(max(all_levels)))

    return sorted(set(selected), reverse=True)


def build_real_xlabels(n_order: list[float]) -> list[str]:
    return [f"{(v / 100.0) * N_FULL_RATE_KG_HA:.0f} kg" for v in n_order]


def build_real_ylabels(stats_scenario: pd.DataFrame, irr_order: list[float]) -> list[str]:
    irr_mm = stats_scenario.groupby("Irr_pct")["I_mean"].mean().reindex(irr_order)
    return ["NA" if pd.isna(v) else f"{v:.0f} mm" for v in irr_mm.values]


def draw_heatmap(
    ax,
    M: np.ndarray,
    title: str,
    cmap: str,
    fmt: str,
    xlabels: list[str],
    ylabels: list[str],
    feasible_mask: np.ndarray,
    optimum_pos: tuple[int, int] | None,
    xlabel_text: str,
    ylabel_text: str,
    vmin: float | None = None,
    vmax: float | None = None,
):
    im = ax.imshow(M, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)

    nrows, ncols = M.shape
    ax.set_title(title, pad=3, fontsize=9)
    ax.set_xticks(list(range(ncols)))
    ax.set_yticks(list(range(nrows)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels)
    ax.set_xlabel(xlabel_text)
    ax.set_ylabel(ylabel_text)

    add_grid(ax, nrows=nrows, ncols=ncols)

    fs_val = 6.3 if max(nrows, ncols) <= 8 else 5.8
    fs_nf = 5.0

    for r in range(nrows):
        for c in range(ncols):
            v = M[r, c]
            txt = "NA" if np.isnan(v) else fmt.format(v)
            is_feasible = bool(feasible_mask[r, c]) if not np.isnan(v) else True

            y_val = r - 0.12 if (not is_feasible and not np.isnan(v)) else r
            ax.text(c, y_val, txt, ha="center", va="center", fontsize=fs_val, color=TEXT_COLOR, zorder=4)

            if (not np.isnan(v)) and (not is_feasible):
                ax.text(
                    c,
                    r + 0.24,
                    NF_TEXT,
                    ha="center",
                    va="center",
                    fontsize=fs_nf,
                    fontweight="bold",
                    color=TEXT_COLOR,
                    zorder=6,
                )

    if optimum_pos is not None:
        ro, co = optimum_pos
        ax.add_patch(
            Rectangle(
                (co - 0.5, ro - 0.5),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=2.0,
                zorder=7,
            )
        )

    return im


def draw_irrigation_bar(ax, stats_scenario: pd.DataFrame, irr_order: list[float], xlabels: list[str], xlabel_text: str, title: str) -> None:
    means = stats_scenario.groupby("Irr_pct")["I_mean"].mean().reindex(irr_order)
    xs = np.arange(len(irr_order))

    ax.bar(xs, means.values, width=0.75)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_xlabel(xlabel_text)
    ax.set_ylabel("Seasonal irrigation (mm)")
    ax.set_title(title, pad=3, fontsize=9)
    ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.7)


def get_orders(stats: pd.DataFrame, scenarios: list[str]) -> tuple[list[float], list[float]]:
    irr_all = sorted(stats["Irr_pct"].dropna().unique().tolist(), reverse=True)
    n_all = sorted(stats["N_pct"].dropna().unique().tolist(), reverse=True)

    optimum_irr = []
    optimum_n = []
    for scenario in scenarios:
        sc_df = stats[stats["Scenario"] == scenario].copy()
        if sc_df.empty:
            continue
        opt = compute_optimum(sc_df)
        optimum_irr.append(float(opt["Irr_pct"]))
        optimum_n.append(float(opt["N_pct"]))

    irr_order = select_levels_with_optimum(irr_all, DISPLAY_LEVELS_IRR, optimum_irr)
    n_order = select_levels_with_optimum(n_all, DISPLAY_LEVELS_N, optimum_n)
    return irr_order, n_order


# ============================================================
# FIGURES HEATMAPS
# ============================================================
def build_parameter_heatmap_figure(
    stats: pd.DataFrame,
    scenarios: list[str],
    value_col: str,
    cmap: str,
    cbar_label: str,
    fmt: str,
    figure_title: str,
    out_png: str,
    use_real_value_axes: bool = False,
) -> None:
    irr_order, n_order = get_orders(stats, scenarios)

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.8))
    axes = axes.flatten()

    all_vals = stats[value_col].dropna()
    vmin = float(all_vals.min()) if not all_vals.empty else None
    vmax = float(all_vals.max()) if not all_vals.empty else None

    last_im = None

    for ax, scenario in zip(axes, scenarios):
        sc_df = stats[
            (stats["Scenario"] == scenario)
            & (stats["Irr_pct"].isin(irr_order))
            & (stats["N_pct"].isin(n_order))
        ].copy()

        if sc_df.empty:
            ax.axis("off")
            ax.set_title(f"{scenario}\n(no data)", pad=3, fontsize=9)
            continue

        M = matrix_from_stats(sc_df, value_col, irr_order, n_order)
        M_Feasible = matrix_bool_from_stats(sc_df, "Feasible", irr_order, n_order)

        opt = compute_optimum(sc_df)
        r_opt = irr_order.index(float(opt["Irr_pct"]))
        c_opt = n_order.index(float(opt["N_pct"]))
        opt_pos = (r_opt, c_opt)

        if use_real_value_axes:
            xlabels = build_real_xlabels(n_order)
            ylabels = build_real_ylabels(sc_df, irr_order)
            xlabel_text = "Applied N rate"
            ylabel_text = "Seasonal irrigation"
        else:
            xlabels = [f"{v:g}% N" for v in n_order]
            ylabels = [f"{v:g}% ETc" for v in irr_order]
            xlabel_text = "Nitrogen treatment"
            ylabel_text = "Irrigation treatment"

        last_im = draw_heatmap(
            ax=ax,
            M=M,
            title=scenario,
            cmap=cmap,
            fmt=fmt,
            xlabels=xlabels,
            ylabels=ylabels,
            feasible_mask=M_Feasible,
            optimum_pos=opt_pos,
            xlabel_text=xlabel_text,
            ylabel_text=ylabel_text,
            vmin=vmin,
            vmax=vmax,
        )

    fig.suptitle(figure_title, y=0.97, fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.90, top=0.91, bottom=0.12, wspace=0.25, hspace=0.35)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.tolist(), fraction=0.025, pad=0.02)
        cbar.set_label(cbar_label, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_png}")


# ============================================================
# FIGURE IRRIGATION
# ============================================================
def build_irrigation_figure(
    stats: pd.DataFrame,
    scenarios: list[str],
    figure_title: str,
    out_png: str,
    use_real_value_axes: bool = False,
) -> None:
    irr_order, _ = get_orders(stats, scenarios)

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.8))
    axes = axes.flatten()

    for ax, scenario in zip(axes, scenarios):
        sc_df = stats[
            (stats["Scenario"] == scenario)
            & (stats["Irr_pct"].isin(irr_order))
        ].copy()

        if sc_df.empty:
            ax.axis("off")
            ax.set_title(f"{scenario}\n(no data)", pad=3, fontsize=9)
            continue

        if use_real_value_axes:
            xlabels = build_real_ylabels(sc_df, irr_order)
            xlabel_text = "Seasonal irrigation class"
        else:
            xlabels = [f"{v:g}% ETc" for v in irr_order]
            xlabel_text = "Irrigation treatment"

        draw_irrigation_bar(
            ax=ax,
            stats_scenario=sc_df,
            irr_order=irr_order,
            xlabels=xlabels,
            xlabel_text=xlabel_text,
            title=scenario,
        )

    fig.suptitle(figure_title, y=0.97, fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.91, bottom=0.12, wspace=0.25, hspace=0.35)
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_png}")


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    if not os.path.exists(SUMMARY_CSV):
        raise FileNotFoundError(f"CSV not found: {SUMMARY_CSV}")

    stats = pd.read_csv(SUMMARY_CSV, encoding="utf-8-sig")

    numeric_cols = [
        "Irr_pct", "N_pct", "Y_mean", "Y_rainfed_mean", "DeltaY_mean",
        "I_mean", "IWUE_mean_kgm3", "Nres_mean", "Yield_ratio_to_Nmax",
    ]
    for c in numeric_cols:
        if c in stats.columns:
            stats[c] = pd.to_numeric(stats[c], errors="coerce")

    if "Feasible" not in stats.columns:
        raise ValueError("Column 'Feasible' is missing from the summary CSV.")

    stats = stats.dropna(subset=["Scenario", "Irr_pct", "N_pct"]).copy()
    if stats.empty:
        raise ValueError("Summary CSV is empty or invalid.")

    scenarios_present = [s for s in SCENARIOS_TO_KEEP if s in stats["Scenario"].unique().tolist()]
    if not scenarios_present:
        raise ValueError("Requested projection scenarios are not present in the CSV.")

    # Percentage-axis figures
    build_parameter_heatmap_figure(
        stats=stats,
        scenarios=scenarios_present,
        value_col="Y_mean",
        cmap="YlGn",
        cbar_label="t ha$^{-1}$",
        fmt="{:.2f}",
        figure_title="Projection - Final biomass at harvest, Y (t ha$^{-1}$)",
        out_png=OUT_BIOMASS_PNG,
        use_real_value_axes=False,
    )

    build_irrigation_figure(
        stats=stats,
        scenarios=scenarios_present,
        figure_title="Projection - Seasonal irrigation, I (mm)",
        out_png=OUT_IRRIGATION_PNG,
        use_real_value_axes=False,
    )

    build_parameter_heatmap_figure(
        stats=stats,
        scenarios=scenarios_present,
        value_col="IWUE_mean_kgm3",
        cmap="PuBuGn",
        cbar_label="kg m$^{-3}$",
        fmt="{:.2f}",
        figure_title="Projection - Irrigation water use efficiency, IWUE (kg m$^{-3}$)",
        out_png=OUT_IWUE_PNG,
        use_real_value_axes=False,
    )

    build_parameter_heatmap_figure(
        stats=stats,
        scenarios=scenarios_present,
        value_col="Nres_mean",
        cmap="YlOrRd",
        cbar_label="kg N ha$^{-1}$",
        fmt="{:.1f}",
        figure_title="Projection - Residual mineral N at harvest (kg N ha$^{-1}$)",
        out_png=OUT_NRES_PNG,
        use_real_value_axes=False,
    )

    # Real-value-axis figures
    build_parameter_heatmap_figure(
        stats=stats,
        scenarios=scenarios_present,
        value_col="Y_mean",
        cmap="YlGn",
        cbar_label="t ha$^{-1}$",
        fmt="{:.2f}",
        figure_title="Projection - Final biomass at harvest, Y",
        out_png=OUT_BIOMASS_VALUES_PNG,
        use_real_value_axes=True,
    )

    build_irrigation_figure(
        stats=stats,
        scenarios=scenarios_present,
        figure_title="Projection - Seasonal irrigation, I",
        out_png=OUT_IRRIGATION_VALUES_PNG,
        use_real_value_axes=True,
    )

    build_parameter_heatmap_figure(
        stats=stats,
        scenarios=scenarios_present,
        value_col="IWUE_mean_kgm3",
        cmap="PuBuGn",
        cbar_label="kg m$^{-3}$",
        fmt="{:.2f}",
        figure_title="Projection - IWUE",
        out_png=OUT_IWUE_VALUES_PNG,
        use_real_value_axes=True,
    )

    build_parameter_heatmap_figure(
        stats=stats,
        scenarios=scenarios_present,
        value_col="Nres_mean",
        cmap="YlOrRd",
        cbar_label="kg N ha$^{-1}$",
        fmt="{:.1f}",
        figure_title="Projection - Residual mineral N at harvest",
        out_png=OUT_NRES_VALUES_PNG,
        use_real_value_axes=True,
    )


if __name__ == "__main__":
    main()
