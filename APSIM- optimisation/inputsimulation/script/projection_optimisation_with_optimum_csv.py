# -*- coding: utf-8 -*-
# Projection minimale : 4 figures globales uniquement (sans CSV)

import os
import re
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================================
# CONFIGURATION
# ============================================================
DB_DIR = os.environ.get(
    "DB_DIR",
    "/srv/lustre01/project/assiwat-tbwgr0oduuk/users/mohamed.benaly/outputsimulation/dbfiels",
)
OUT_DIR = os.environ.get(
    "OUT_DIR",
    "/srv/lustre01/project/assiwat-tbwgr0oduuk/users/mohamed.benaly/outputsimulation/csvetfigures/",
)
os.makedirs(OUT_DIR, exist_ok=True)

DB_PATHS = sorted(
    [os.path.join(DB_DIR, f) for f in os.listdir(DB_DIR) if f.endswith(".db")]
)
if not DB_PATHS:
    raise FileNotFoundError(f"Aucun fichier .db trouvé dans {DB_DIR}")

DATE_COL = "Clock.Today"
IRR_COL = "Irrigation.IrrigationApplied"
STAGE_COL = "Maize.Phenology.CurrentStageName"
PREFERRED_YIELD_COL = "ABGB"
PREFERRED_NRES_COL = "Nres_kg_ha"

SOWING_MONTH = 3
SOWING_DAY = 23

SCENARIOS_TO_KEEP = [
    "SSP245 - 2050",
    "SSP245 - 2100",
    "SSP585 - 2050",
    "SSP585 - 2100",
]

YIELD_SUFFICIENCY_THRESHOLD = 0.95
IWUE_COMPARABLE_TOL_REL = 0.01

OUT_BIOMASS_PNG = os.path.join(OUT_DIR, "projection_biomass_all_scenarios.png")
OUT_IRRIGATION_PNG = os.path.join(OUT_DIR, "projection_irrigation_all_scenarios.png")
OUT_IWUE_PNG = os.path.join(OUT_DIR, "projection_IWUE_all_scenarios.png")
OUT_NRES_PNG = os.path.join(OUT_DIR, "projection_Nres_all_scenarios.png")
OUT_OPTIMUM_CSV = os.path.join(OUT_DIR, "projection_optimum_summary_merged_4db_incrementalIWUE.csv")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 16,
})

TEXT_COLOR = "black"
NF_TEXT = "(NF)"
NF_Y_OFFSET = 0.28
NF_X_OFFSET = 0.00


# ============================================================
# HELPERS EXTRACTION
# ============================================================
def get_report_columns(con: sqlite3.Connection) -> list[str]:
    return pd.read_sql_query("PRAGMA table_info('Report')", con)["name"].tolist()


def parse_scenario(name: str) -> str | None:
    s = str(name)
    if ("Baseline" in s) and ("2000" in s):
        return "2000 (Baseline)"
    if "SSP245-2050" in s:
        return "SSP245 - 2050"
    if "SSP245-2100" in s:
        return "SSP245 - 2100"
    if "SSP585-2050" in s:
        return "SSP585 - 2050"
    if "SSP585-2100" in s:
        return "SSP585 - 2100"
    return None


def parse_model_tag(sim_name: str) -> str:
    s = str(sim_name)
    m = re.search(r"\b(M\d+)\b", s)
    if m:
        return m.group(1)
    m = re.search(r"\bModel\s*(\d+)\b", s, flags=re.IGNORECASE)
    if m:
        return f"M{m.group(1)}"
    m = re.search(r"\bGCM\s*(\d+)\b", s, flags=re.IGNORECASE)
    if m:
        return f"M{m.group(1)}"
    return "M?"


def pick_col(cols: list[str], preferred: str, contains_key: str, label_for_error: str) -> str:
    if preferred in cols:
        return preferred
    cands = [c for c in cols if contains_key.lower() in c.lower()]
    if cands:
        return cands[0]
    raise ValueError(f"❌ Colonne {label_for_error} introuvable dans Report. Exemple colonnes: {cols[:25]}")


def normalize_level_to_percent(x: float) -> float:
    if pd.isna(x):
        return np.nan
    x = float(x)
    if x <= 1.5:
        x *= 100.0
    x = round(x, 6)
    if abs(x - round(x)) < 1e-6:
        return float(int(round(x)))
    return x


def compute_optimum(stats_scenario: pd.DataFrame) -> pd.Series:
    feasible = stats_scenario[stats_scenario["Feasible"]].copy()
    if feasible.empty:
        raise ValueError("❌ Aucun traitement faisable pour ce scénario.")

    feasible = feasible.dropna(subset=["IWUE_mean_kgm3"]).copy()
    if feasible.empty:
        raise ValueError("❌ Aucun traitement faisable avec IWUE calculable pour ce scénario.")

    iwue_max = feasible["IWUE_mean_kgm3"].max()
    iwue_cutoff = iwue_max * (1 - IWUE_COMPARABLE_TOL_REL)

    comparable = feasible[feasible["IWUE_mean_kgm3"] >= iwue_cutoff].copy()
    comparable = comparable.sort_values(
        by=["Nres_mean", "IWUE_mean_kgm3", "N_pct", "Irr_pct"],
        ascending=[True, False, True, True],
    )
    return comparable.iloc[0]


def export_optimum_csv(stats: pd.DataFrame) -> pd.DataFrame:
    """Crée un petit CSV résumé des optimums de projection avec les mêmes colonnes que baseline."""
    scenario_labels = {
        "SSP245 - 2050": "ssp245-2050",
        "SSP245 - 2100": "ssp245-2100",
        "SSP585 - 2050": "ssp585-2050",
        "SSP585 - 2100": "ssp585-2100",
    }

    rows = []
    for scenario in SCENARIOS_TO_KEEP:
        sc_df = stats[stats["Scenario"] == scenario].copy()
        if sc_df.empty:
            continue

        optimum = compute_optimum(sc_df)
        rows.append({
            "Scenario": scenario_labels.get(scenario, scenario),
            "Nrest": round(float(optimum["Nres_mean"]), 2),
            "IWUE": round(float(optimum["IWUE_mean_kgm3"]), 2),
            "Rndt": round(float(optimum["Y_mean"]), 2),
            "Etc%": int(round(float(optimum["Irr_pct"]))),
            "N%": int(round(float(optimum["N_pct"]))),
            "SeasonalIrrigation_mm": round(float(optimum["I_mean"]), 1),
            "Y_rainfed": round(float(optimum["Y_rainfed_mean"]), 2),
            "DeltaY": round(float(optimum["DeltaY_mean"]), 2),
            "Yield_ratio_to_Nmax": round(float(optimum["Yield_ratio_to_Nmax"]), 3),
            "Feasible": bool(optimum["Feasible"]),
        })

    optimum_df = pd.DataFrame(rows)
    optimum_df.to_csv(OUT_OPTIMUM_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ CSV optimum enregistré : {OUT_OPTIMUM_CSV}")
    return optimum_df


def extract_one_db(db_path: str) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print(f"Lecture DB : {db_path}")

    con = sqlite3.connect(db_path)
    cols = get_report_columns(con)

    for must in [DATE_COL, STAGE_COL]:
        if must not in cols:
            raise ValueError(f"❌ Colonne '{must}' introuvable dans Report de {db_path}")

    if IRR_COL in cols:
        irr_col = IRR_COL
    else:
        irr_cands = [c for c in cols if "irrig" in c.lower()]
        if not irr_cands:
            raise ValueError(f"❌ Aucune colonne irrigation trouvée dans {db_path}")
        pref = [c for c in irr_cands if "appl" in c.lower()]
        irr_col = pref[0] if pref else irr_cands[0]

    yield_col = pick_col(cols, PREFERRED_YIELD_COL, "abgb", "Yield/ABGB")
    nres_col = pick_col(cols, PREFERRED_NRES_COL, "nres", "Nres_kg_ha")

    if "SimulationID" not in cols:
        raise ValueError(f"❌ 'SimulationID' introuvable dans Report de {db_path}")

    q = f'''
    SELECT
        r.SimulationID,
        s.Name as SimulationName,
        r."{DATE_COL}"  as TheDate,
        r."{irr_col}"   as Irr,
        r."{STAGE_COL}" as StageName,
        r."{yield_col}" as Y_final_raw,
        r."{nres_col}"  as Nres_kg_ha
    FROM Report r
    JOIN _Simulations s ON s.ID = r.SimulationID
    '''

    df = pd.read_sql_query(q, con)
    con.close()

    df["DB_Source"] = os.path.basename(db_path)
    df["TheDate"] = pd.to_datetime(df["TheDate"], errors="coerce")
    df = df.dropna(subset=["TheDate"]).copy()
    df["_rowid"] = np.arange(len(df), dtype=int)

    df["Irr"] = pd.to_numeric(df["Irr"], errors="coerce").fillna(0.0)
    df["Y_final_raw"] = pd.to_numeric(df["Y_final_raw"], errors="coerce")
    df["Nres_kg_ha"] = pd.to_numeric(df["Nres_kg_ha"], errors="coerce")
    df["StageName"] = df["StageName"].astype(str).str.strip()

    sim = df["SimulationName"].astype(str).str.replace(",", ".", regex=False)
    num = r"([0-9]*\.?[0-9]+)"

    df["N_raw"] = sim.str.extract(rf"(?:Fertiliser|Fertilizer)\s*[:=_-]?\s*{num}", expand=False)
    df["Irr_raw"] = sim.str.extract(rf"Strategy\s*[:=_-]?\s*{num}\s*Etc", expand=False)

    df = df.dropna(subset=["N_raw", "Irr_raw"]).copy()
    df["N_pct"] = pd.to_numeric(df["N_raw"], errors="coerce")
    df["Irr_pct"] = pd.to_numeric(df["Irr_raw"], errors="coerce")
    df = df.dropna(subset=["N_pct", "Irr_pct"]).copy()

    df["N_pct"] = df["N_pct"].apply(normalize_level_to_percent)
    df["Irr_pct"] = df["Irr_pct"].apply(normalize_level_to_percent)

    df["Scenario"] = df["SimulationName"].apply(parse_scenario)
    df = df.dropna(subset=["Scenario"]).copy()
    df = df[df["Scenario"].isin(SCENARIOS_TO_KEEP)].copy()

    if df.empty:
        print(f"⚠️ Aucune ligne de projection trouvée dans {db_path}")
        return pd.DataFrame()

    df["ModelTag"] = df["SimulationName"].apply(parse_model_tag)

    grp_keys = ["DB_Source", "SimulationID", "Scenario", "ModelTag", "Irr_pct", "N_pct"]
    df = df.sort_values(grp_keys + ["TheDate", "_rowid"]).reset_index(drop=True)

    sow = df[df["StageName"].eq("Sowing")].copy()
    sow = sow[(sow["TheDate"].dt.month == SOWING_MONTH) & (sow["TheDate"].dt.day == SOWING_DAY)].copy()
    if sow.empty:
        print(f"⚠️ Aucun semis trouvé à la date configurée dans {db_path}")
        return pd.DataFrame()

    sow = sow[grp_keys + ["TheDate"]].drop_duplicates().rename(columns={"TheDate": "SowingDate"})
    sow["SowingYear"] = sow["SowingDate"].dt.year.astype(int)

    harv = df[df["StageName"].eq("HarvestRipe")].copy()
    harv = harv[grp_keys + ["TheDate", "_rowid"]].rename(columns={"TheDate": "HarvestRipeDate"})
    harv = harv.sort_values(grp_keys + ["HarvestRipeDate", "_rowid"])

    matched = []
    for key, g_sow in sow.groupby(grp_keys):
        g_harv = harv[
            (harv["DB_Source"] == key[0])
            & (harv["SimulationID"] == key[1])
            & (harv["Scenario"] == key[2])
            & (harv["ModelTag"] == key[3])
            & (harv["Irr_pct"] == key[4])
            & (harv["N_pct"] == key[5])
        ]
        if g_harv.empty:
            continue

        harv_dates = g_harv["HarvestRipeDate"].values
        for _, row in g_sow.iterrows():
            sd = row["SowingDate"].to_datetime64()
            idx = np.searchsorted(harv_dates, sd, side="right")
            if idx < len(harv_dates):
                matched.append(
                    {
                        "DB_Source": key[0],
                        "SimulationID": key[1],
                        "Scenario": key[2],
                        "ModelTag": key[3],
                        "Irr_pct": key[4],
                        "N_pct": key[5],
                        "SowingDate": row["SowingDate"],
                        "SowingYear": row["SowingYear"],
                        "HarvestRipeDate": pd.Timestamp(harv_dates[idx]),
                    }
                )

    windows = pd.DataFrame(matched)
    if windows.empty:
        print(f"⚠️ Aucun couple semis-récolte trouvé dans {db_path}")
        return pd.DataFrame()

    df_small = df[grp_keys + ["TheDate", "Irr", "Y_final_raw", "Nres_kg_ha", "StageName", "_rowid"]].copy()

    seasonal_rows = []
    for key, g in df_small.groupby(grp_keys):
        g = g.sort_values(["TheDate", "_rowid"])

        w = windows[
            (windows["DB_Source"] == key[0])
            & (windows["SimulationID"] == key[1])
            & (windows["Scenario"] == key[2])
            & (windows["ModelTag"] == key[3])
            & (windows["Irr_pct"] == key[4])
            & (windows["N_pct"] == key[5])
        ]
        if w.empty:
            continue

        for _, wr in w.iterrows():
            sd = wr["SowingDate"]
            hd = wr["HarvestRipeDate"]

            mask = (g["TheDate"] >= sd) & (g["TheDate"] <= hd)
            I_season = g.loc[mask, "Irr"].sum()

            g_h = (
                g[(g["StageName"] == "HarvestRipe") & (g["TheDate"] == hd)]
                .sort_values(["TheDate", "_rowid"])
                .head(1)
            )
            if g_h.empty:
                g_h = (
                    g[(g["StageName"] == "HarvestRipe") & (g["TheDate"] > sd)]
                    .sort_values(["TheDate", "_rowid"])
                    .head(1)
                )
            if g_h.empty:
                continue

            seasonal_rows.append(
                {
                    "DB_Source": key[0],
                    "SimulationID": key[1],
                    "Scenario": key[2],
                    "ModelTag": key[3],
                    "Irr_pct": float(key[4]),
                    "N_pct": float(key[5]),
                    "SowingYear": int(wr["SowingYear"]),
                    "SowingDate": sd,
                    "HarvestRipeDate": hd,
                    "SeasonalIrrigation": float(I_season),
                    "Y_final": float(pd.to_numeric(g_h["Y_final_raw"].iloc[0], errors="coerce")),
                    "Nres_final": float(pd.to_numeric(g_h["Nres_kg_ha"].iloc[0], errors="coerce")),
                }
            )

    seasonal = pd.DataFrame(seasonal_rows)
    if seasonal.empty:
        print(f"⚠️ Aucune métrique saisonnière calculée dans {db_path}")
        return pd.DataFrame()

    return seasonal


# ============================================================
# HELPERS FIGURES
# ============================================================
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
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)


def draw_heatmap(ax, M: np.ndarray, title: str, cmap: str, fmt: str, xlabels: list[str], ylabels: list[str], feasible_mask: np.ndarray, optimum_pos: tuple[int, int] | None, show_opt_label: bool):
    nrows, ncols = M.shape
    im = ax.imshow(M, cmap=cmap, aspect="equal")

    ax.set_title(title, pad=4)
    ax.set_xticks(list(range(ncols)))
    ax.set_yticks(list(range(nrows)))
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Nitrogen treatment")
    ax.set_ylabel("Irrigation treatment")

    if ncols > 8:
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha("right")

    add_grid(ax, nrows=nrows, ncols=ncols)

    cell_max = max(nrows, ncols)
    fs = 10 if cell_max <= 7 else (8 if cell_max <= 11 else 6)

    for r in range(nrows):
        for c in range(ncols):
            v = M[r, c]
            txt = "NA" if np.isnan(v) else fmt.format(v)
            ax.text(c, r, txt, ha="center", va="center", fontsize=fs, color=TEXT_COLOR, zorder=4)
            if (not np.isnan(v)) and (not bool(feasible_mask[r, c])):
                ax.text(
                    c + NF_X_OFFSET,
                    r + NF_Y_OFFSET,
                    NF_TEXT,
                    ha="center",
                    va="center",
                    fontsize=max(6, fs - 2),
                    fontweight="bold",
                    color=TEXT_COLOR,
                    zorder=6,
                )

    if optimum_pos is not None:
        ro, co = optimum_pos
        ax.add_patch(Rectangle((co - 0.5, ro - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2.8, zorder=7))
        if show_opt_label:
            ax.text(
                co + 0.24,
                ro + 0.30,
                "Opt",
                fontsize=7,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="black", linewidth=0.6),
                zorder=8,
            )

    return im


def draw_irrigation_bar(ax, stats: pd.DataFrame, irr_order: list[float], title: str) -> None:
    means = stats.groupby("Irr_pct")["I_mean"].mean().reindex(irr_order)
    xs = np.arange(len(irr_order))
    ax.bar(xs, means.values)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{v:g}% ETc" for v in irr_order],
        rotation=45 if len(irr_order) > 8 else 0,
        ha="right" if len(irr_order) > 8 else "center",
    )
    ax.set_ylabel("Seasonal irrigation (mm)")
    ax.set_title(title, pad=4)
    ax.grid(axis="y", linestyle="-", linewidth=0.4)


def build_parameter_heatmap_figure(stats: pd.DataFrame, scenarios: list[str], value_col: str, cmap: str, cbar_label: str, fmt: str, figure_title: str, out_png: str, show_opt_label_on_panels: bool = False) -> None:
    irr_order = sorted(stats["Irr_pct"].dropna().unique().tolist(), reverse=True)
    n_order = sorted(stats["N_pct"].dropna().unique().tolist(), reverse=True)
    xlabels = [f"{v:g}% N" for v in n_order]
    ylabels = [f"{v:g}% ETc" for v in irr_order]

    fig_w = min(26, 14.0 + max(0, (len(n_order) - 3)) * 0.55)
    fig_h = min(22, 13.0 + max(0, (len(irr_order) - 3)) * 0.35)
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), constrained_layout=False)
    axes = axes.flatten()
    last_im = None

    for ax, scenario in zip(axes, scenarios):
        sc_df = stats[stats["Scenario"] == scenario].copy()
        if sc_df.empty:
            ax.axis("off")
            ax.set_title(f"{scenario}\n(no data)")
            continue

        M = matrix_from_stats(sc_df, value_col, irr_order=irr_order, n_order=n_order)
        M_Feasible = matrix_bool_from_stats(sc_df, "Feasible", irr_order=irr_order, n_order=n_order)

        opt_pos = None
        try:
            optimum = compute_optimum(sc_df)
            r_opt = irr_order.index(float(optimum["Irr_pct"]))
            c_opt = n_order.index(float(optimum["N_pct"]))
            opt_pos = (r_opt, c_opt)
        except Exception:
            opt_pos = None

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
            show_opt_label=show_opt_label_on_panels,
        )

    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.08, top=0.92, wspace=0.02, hspace=0.15)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.tolist(), fraction=0.020, pad=0.008)
        cbar.set_label(cbar_label)

    fig.suptitle(figure_title, y=0.96)
    footer = (
        f"Optimum computed per scenario: max IWUE among treatments meeting Y/Ymax (same ETc) ≥ {YIELD_SUFFICIENCY_THRESHOLD:.2f}; "
        f"ties within {IWUE_COMPARABLE_TOL_REL*100:.0f}% resolved by minimum residual N. "
        f"(NF) = non-feasible."
    )
    fig.text(0.5, 0.01, footer, ha="center", va="top", fontsize=9)
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figure enregistrée : {out_png}")


def build_irrigation_figure(stats: pd.DataFrame, scenarios: list[str], figure_title: str, out_png: str) -> None:
    irr_order = sorted(stats["Irr_pct"].dropna().unique().tolist(), reverse=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=False)
    axes = axes.flatten()
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.09, top=0.92, wspace=0.06, hspace=0.28)

    for ax, scenario in zip(axes, scenarios):
        sc_df = stats[stats["Scenario"] == scenario].copy()
        if sc_df.empty:
            ax.axis("off")
            ax.set_title(f"{scenario}\n(no data)")
            continue
        draw_irrigation_bar(ax, sc_df, irr_order=irr_order, title=scenario)

    fig.suptitle(figure_title, y=0.96)
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Figure enregistrée : {out_png}")


# ============================================================
# MAIN
# ============================================================
def build_summary_stats() -> pd.DataFrame:
    all_seasonal = []
    for db_path in DB_PATHS:
        seasonal_db = extract_one_db(db_path)
        if not seasonal_db.empty:
            all_seasonal.append(seasonal_db)

    if not all_seasonal:
        raise ValueError("❌ Aucune donnée valide extraite depuis les DB.")

    seasonal = pd.concat(all_seasonal, ignore_index=True)
    seasonal = seasonal.drop_duplicates(
        subset=["DB_Source", "Scenario", "ModelTag", "Irr_pct", "N_pct", "SowingYear", "SowingDate", "HarvestRipeDate"]
    ).copy()

    rainfed = seasonal.loc[
        seasonal["Irr_pct"] == 0,
        ["Scenario", "ModelTag", "SowingYear", "N_pct", "Y_final"],
    ].copy()
    rainfed = (
        rainfed.groupby(["Scenario", "ModelTag", "SowingYear", "N_pct"], as_index=False)
        .agg(Y_rainfed=("Y_final", "mean"))
    )

    seasonal = seasonal.merge(
        rainfed,
        on=["Scenario", "ModelTag", "SowingYear", "N_pct"],
        how="left",
    )

    seasonal["DeltaY"] = (seasonal["Y_final"] - seasonal["Y_rainfed"]).clip(lower=0)
    seasonal["IWUE_tha_mm"] = np.where(
        seasonal["SeasonalIrrigation"] > 0,
        seasonal["DeltaY"] / seasonal["SeasonalIrrigation"],
        np.nan,
    )
    seasonal["IWUE_kg_m3"] = seasonal["IWUE_tha_mm"] * 100.0

    by_model = (
        seasonal.groupby(["Scenario", "ModelTag", "Irr_pct", "N_pct"], as_index=False)
        .agg(
            Y_model_mean=("Y_final", "mean"),
            Yrf_model_mean=("Y_rainfed", "mean"),
            DeltaY_model_mean=("DeltaY", "mean"),
            I_model_mean=("SeasonalIrrigation", "mean"),
            IWUE_model_mean=("IWUE_kg_m3", "mean"),
            Nres_model_mean=("Nres_final", "mean"),
            n_years=("SowingYear", "nunique"),
        )
    )

    stats = (
        by_model.groupby(["Scenario", "Irr_pct", "N_pct"], as_index=False)
        .agg(
            Y_mean=("Y_model_mean", "mean"),
            Y_std=("Y_model_mean", "std"),
            Y_rainfed_mean=("Yrf_model_mean", "mean"),
            DeltaY_mean=("DeltaY_model_mean", "mean"),
            I_mean=("I_model_mean", "mean"),
            I_std=("I_model_mean", "std"),
            IWUE_mean_kgm3=("IWUE_model_mean", "mean"),
            IWUE_std_kgm3=("IWUE_model_mean", "std"),
            Nres_mean=("Nres_model_mean", "mean"),
            Nres_std=("Nres_model_mean", "std"),
            n_models_total=("Irr_pct", "count"),
            mean_years_per_model=("n_years", "mean"),
        )
    )

    for c in ["Y_std", "I_std", "IWUE_std_kgm3", "Nres_std"]:
        stats[c] = stats[c].fillna(0.0)

    nmax = (
        stats.groupby(["Scenario", "Irr_pct"], as_index=False)["N_pct"]
        .max()
        .rename(columns={"N_pct": "N_max"})
    )
    stats = stats.merge(nmax, on=["Scenario", "Irr_pct"], how="left")

    y_nmax = (
        stats.loc[stats["N_pct"] == stats["N_max"], ["Scenario", "Irr_pct", "Y_mean"]]
        .drop_duplicates(subset=["Scenario", "Irr_pct"])
        .rename(columns={"Y_mean": "Y_Nmax_sameIrr"})
    )
    stats = stats.merge(y_nmax, on=["Scenario", "Irr_pct"], how="left")

    if stats["Y_Nmax_sameIrr"].isna().any():
        stats["Y_Nmax_sameIrr"] = stats["Y_Nmax_sameIrr"].fillna(
            stats.groupby(["Scenario", "Irr_pct"])["Y_mean"].transform("max")
        )

    stats["Yield_ratio_to_Nmax"] = stats["Y_mean"] / stats["Y_Nmax_sameIrr"].replace(0, np.nan)
    stats["Feasible"] = (stats["Yield_ratio_to_Nmax"] >= YIELD_SUFFICIENCY_THRESHOLD).fillna(False).astype(bool)

    stats = stats.sort_values(["Scenario", "Irr_pct", "N_pct"], ascending=[True, False, False]).reset_index(drop=True)
    return stats


def build_figures(stats: pd.DataFrame) -> None:
    scenarios_present = [s for s in SCENARIOS_TO_KEEP if s in stats["Scenario"].unique().tolist()]
    if not scenarios_present:
        raise ValueError("❌ Aucun des scénarios demandés n'est présent dans les données.")

    print("\n" + "=" * 80)
    print("TRACÉ DES FIGURES DE PROJECTION (SANS CSV)")
    print("=" * 80)

    build_parameter_heatmap_figure(
        stats=stats,
        scenarios=scenarios_present,
        value_col="Y_mean",
        cmap="YlGn",
        cbar_label="t ha$^{-1}$",
        fmt="{:.2f}",
        figure_title="Projection – Final biomass at harvest, Y (t ha$^{-1}$)",
        out_png=OUT_BIOMASS_PNG,
        show_opt_label_on_panels=False,
    )

    build_irrigation_figure(
        stats=stats,
        scenarios=scenarios_present,
        figure_title="Projection – Seasonal irrigation, I (mm)",
        out_png=OUT_IRRIGATION_PNG,
    )

    build_parameter_heatmap_figure(
        stats=stats,
        scenarios=scenarios_present,
        value_col="IWUE_mean_kgm3",
        cmap="PuBuGn",
        cbar_label="kg m$^{-3}$",
        fmt="{:.2f}",
        figure_title="Projection irrigation water use efficiency, IWUE (kg m$^{-3}$)",
        out_png=OUT_IWUE_PNG,
        show_opt_label_on_panels=True,
    )

    build_parameter_heatmap_figure(
        stats=stats,
        scenarios=scenarios_present,
        value_col="Nres_mean",
        cmap="YlOrRd",
        cbar_label="kg N ha$^{-1}$",
        fmt="{:.1f}",
        figure_title="Projection – Residual mineral N at harvest (kg N ha$^{-1}$)",
        out_png=OUT_NRES_PNG,
        show_opt_label_on_panels=False,
    )


def main() -> None:
    stats = build_summary_stats()
    build_figures(stats)


if __name__ == "__main__":
    main()