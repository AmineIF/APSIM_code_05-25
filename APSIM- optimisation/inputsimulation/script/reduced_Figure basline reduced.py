# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ============================================================
# CONFIGURATION
# ============================================================
SUMMARY_CSV = "/srv/lustre01/project/assiwat-tbwgr0oduuk/users/mohamed.benaly/outputsimulation/csvetfigures/baseline_summary_from_detailcsv_merged_4db_incrementalIWUE_CORRECT.csv"
OUT_FIG_PNG = "/srv/lustre01/project/assiwat-tbwgr0oduuk/users/mohamed.benaly/outputsimulation/csvetfigures/baseline_figure_reduced_clear_article.png"
OUT_FIG_VALUES_PNG = "/srv/lustre01/project/assiwat-tbwgr0oduuk/users/mohamed.benaly/outputsimulation/csvetfigures/baseline_figure_reduced_clear_article_values.png"

YIELD_SUFFICIENCY_THRESHOLD = 0.95
IWUE_COMPARABLE_TOL_REL = 0.01

# Niveaux affichés dans la figure
DISPLAY_LEVELS_IRR = [100, 85, 80, 70, 50, 30, 0]
DISPLAY_LEVELS_N   = [100, 85, 80, 70, 50, 30, 0]

# Dose d'azote correspondant à 100% N
N_FULL_RATE_KG_HA = 210.0

# Style adapté à une figure d'article
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 10,
    "figure.titlesize": 14,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

TEXT_COLOR = "black"
NF_TEXT = "(NF)"


# ============================================================
# HELPERS
# ============================================================
def compute_optimum(stats: pd.DataFrame) -> pd.Series:
    feasible = stats[stats["Feasible"]].copy()
    if feasible.empty:
        raise ValueError("❌ Aucun traitement faisable.")

    feasible = feasible.dropna(subset=["IWUE_mean_kgm3"]).copy()
    if feasible.empty:
        raise ValueError("❌ Aucun traitement faisable avec IWUE calculable.")

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


def select_levels_with_optimum(all_levels: list[float], preferred_levels: list[int], optimum_level: float) -> list[float]:
    all_levels_set = {float(v) for v in all_levels}
    selected = [float(v) for v in preferred_levels if float(v) in all_levels_set]

    selected.append(float(optimum_level))

    if all_levels:
        selected.append(float(min(all_levels)))
        selected.append(float(max(all_levels)))

    return sorted(set(selected), reverse=True)


def build_value_labels(stats_plot: pd.DataFrame, irr_order: list[float], n_order: list[float]) -> tuple[list[str], list[str]]:
    # Axe X: dose d'azote en kg
    xlabels_real = [f"{(v / 100.0) * N_FULL_RATE_KG_HA:.0f} kg" for v in n_order]

    # Axe Y: irrigation saisonnière moyenne réelle en mm
    irr_mm = (
        stats_plot.groupby("Irr_pct")["I_mean"]
        .mean()
        .reindex(irr_order)
    )
    ylabels_real = ["NA" if pd.isna(v) else f"{v:.0f} mm" for v in irr_mm.values]

    return xlabels_real, ylabels_real


def draw_heatmap(
    ax,
    M: np.ndarray,
    title: str,
    cmap: str,
    cbar_label: str,
    fmt: str,
    xlabels: list[str],
    ylabels: list[str],
    feasible_mask: np.ndarray,
    optimum_pos: tuple[int, int] | None,
    xlabel_text: str,
    ylabel_text: str,
) -> None:
    nrows, ncols = M.shape
    im = ax.imshow(M, cmap=cmap, aspect="equal")

    ax.set_title(title, pad=3, fontsize=9)
    ax.set_xticks(list(range(ncols)))
    ax.set_yticks(list(range(nrows)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels)
    ax.set_xlabel(xlabel_text)
    ax.set_ylabel(ylabel_text)

    add_grid(ax, nrows=nrows, ncols=ncols)

    fs_val = 6.8 if max(nrows, ncols) <= 8 else 6.0
    fs_nf = 5.3

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

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def draw_irrigation_bar(ax, stats: pd.DataFrame, irr_order: list[float], xlabels: list[str], xlabel_text: str) -> None:
    means = stats.groupby("Irr_pct")["I_mean"].mean().reindex(irr_order)
    xs = np.arange(len(irr_order))

    ax.bar(xs, means.values, width=0.75)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_xlabel(xlabel_text)
    ax.set_ylabel("Seasonal irrigation (mm)")
    ax.set_title("(b) Seasonal irrigation, I (mm)", pad=3, fontsize=9)
    ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.7)

    divider = make_axes_locatable(ax)
    spacer = divider.append_axes("right", size="5%", pad=0.08)
    spacer.axis("off")


def plot_figure(summary_csv: str, out_png: str, use_real_value_axes: bool = False) -> None:
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f"❌ CSV introuvable : {summary_csv}")

    stats = pd.read_csv(summary_csv, encoding="utf-8-sig")

    numeric_cols = [
        "Irr_pct", "N_pct", "Y_mean", "Y_rainfed_mean", "DeltaY_mean",
        "I_mean", "IWUE_mean_kgm3", "Nres_mean", "Yield_ratio_to_Nmax",
    ]
    for c in numeric_cols:
        if c in stats.columns:
            stats[c] = pd.to_numeric(stats[c], errors="coerce")

    if "Feasible" not in stats.columns:
        raise ValueError("❌ Colonne 'Feasible' absente du CSV résumé.")

    stats = stats.dropna(subset=["Irr_pct", "N_pct"]).copy()
    if stats.empty:
        raise ValueError("❌ Le CSV résumé est vide ou invalide.")

    optimum = compute_optimum(stats)

    irr_all = sorted(stats["Irr_pct"].dropna().unique().tolist(), reverse=True)
    n_all = sorted(stats["N_pct"].dropna().unique().tolist(), reverse=True)

    irr_order = select_levels_with_optimum(
        all_levels=irr_all,
        preferred_levels=DISPLAY_LEVELS_IRR,
        optimum_level=float(optimum["Irr_pct"]),
    )
    n_order = select_levels_with_optimum(
        all_levels=n_all,
        preferred_levels=DISPLAY_LEVELS_N,
        optimum_level=float(optimum["N_pct"]),
    )

    stats_plot = stats[
        stats["Irr_pct"].isin(irr_order) &
        stats["N_pct"].isin(n_order)
    ].copy()

    if stats_plot.empty:
        raise ValueError("❌ Aucun traitement restant après réduction.")

    M_Y = matrix_from_stats(stats_plot, "Y_mean", irr_order, n_order)
    M_IWUE = matrix_from_stats(stats_plot, "IWUE_mean_kgm3", irr_order, n_order)
    M_Nres = matrix_from_stats(stats_plot, "Nres_mean", irr_order, n_order)
    M_Feasible = matrix_bool_from_stats(stats_plot, "Feasible", irr_order, n_order)

    r_opt = irr_order.index(float(optimum["Irr_pct"]))
    c_opt = n_order.index(float(optimum["N_pct"]))
    opt_pos = (r_opt, c_opt)

    if use_real_value_axes:
        xlabels, ylabels = build_value_labels(stats_plot, irr_order, n_order)
        xlabel_text = "Applied N rate"
        ylabel_text = "Seasonal irrigation"
        irrigation_bar_xlabels = ylabels
        irrigation_bar_xlabel_text = "Seasonal irrigation class"
        suptitle = "Baseline irrigation–nitrogen optimization"
    else:
        xlabels = [f"{v:g}% N" for v in n_order]
        ylabels = [f"{v:g}% ETc" for v in irr_order]
        xlabel_text = "Nitrogen treatment"
        ylabel_text = "Irrigation treatment"
        irrigation_bar_xlabels = [f"{v:g}% ETc" for v in irr_order]
        irrigation_bar_xlabel_text = "Irrigation treatment"
        suptitle = "Baseline irrigation–nitrogen optimization"

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8.4, 7.6),
        gridspec_kw={"width_ratios": [1, 1], "height_ratios": [1, 1]},
    )

    draw_heatmap(
        axes[0, 0],
        M_Y,
        "(a) Final biomass at harvest, Y (t ha$^{-1}$)",
        "YlGn",
        "t ha$^{-1}$",
        "{:.2f}",
        xlabels,
        ylabels,
        M_Feasible,
        opt_pos,
        xlabel_text,
        ylabel_text,
    )

    draw_irrigation_bar(
        axes[0, 1],
        stats_plot,
        irr_order,
        irrigation_bar_xlabels,
        irrigation_bar_xlabel_text,
    )

    draw_heatmap(
        axes[1, 0],
        M_IWUE,
        "(c) Irrigation water use efficiency, IWUE (kg m$^{-3}$)",
        "PuBuGn",
        "kg m$^{-3}$",
        "{:.2f}",
        xlabels,
        ylabels,
        M_Feasible,
        opt_pos,
        xlabel_text,
        ylabel_text,
    )

    draw_heatmap(
        axes[1, 1],
        M_Nres,
        "(d) Residual mineral N at harvest (kg N ha$^{-1}$)",
        "YlOrRd",
        "kg N ha$^{-1}$",
        "{:.1f}",
        xlabels,
        ylabels,
        M_Feasible,
        opt_pos,
        xlabel_text,
        ylabel_text,
    )

    fig.suptitle(suptitle, y=0.97, fontsize=11)
    plt.subplots_adjust(left=0.08, right=0.97, top=0.91, bottom=0.11, wspace=0.25, hspace=0.38)
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Figure enregistrée : {out_png}")


if __name__ == "__main__":
    # Figure 1 : axes en pourcentage
    plot_figure(
        summary_csv=SUMMARY_CSV,
        out_png=OUT_FIG_PNG,
        use_real_value_axes=False,
    )

    # Figure 2 : axes en valeurs réelles
    plot_figure(
        summary_csv=SUMMARY_CSV,
        out_png=OUT_FIG_VALUES_PNG,
        use_real_value_axes=True,
    )