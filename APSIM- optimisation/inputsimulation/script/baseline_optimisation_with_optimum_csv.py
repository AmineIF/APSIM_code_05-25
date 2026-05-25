# -*- coding: utf-8 -*-
# Version corrigée :
# 1) extraction depuis les DB
# 2) création d'un CSV détaillé des traitements
# 3) relecture du CSV détaillé
# 4) calcul du résumé depuis ce CSV
# 5) tracé des figures depuis ce CSV

import os
import re
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================================
# CONFIGURATION
# ============================================================
DB_DIR = os.environ.get(
    "DB_DIR",
    "/srv/lustre01/project/assiwat-tbwgr0oduuk/users/mohamed.benaly/outputsimulation/dbfiels"
)
DB_PATHS = sorted(
    [
        os.path.join(DB_DIR, f)
        for f in os.listdir(DB_DIR)
        if f.endswith(".db")
    ]
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
BASELINE_YEARS = range(1985, 2015)
BASELINE_SCENARIO_NAME = "2000 (Baseline)"
YIELD_SUFFICIENCY_THRESHOLD = 0.95
IWUE_COMPARABLE_TOL_REL = 0.01

OUT_DIR = os.environ.get(
    "OUT_DIR",
    "/srv/lustre01/project/assiwat-tbwgr0oduuk/users/mohamed.benaly/outputsimulation/csvetfigures/"
)
os.makedirs(OUT_DIR, exist_ok=True)

# Sorties
SAVE_DETAIL_CSV = True
SAVE_SUMMARY_CSV = True
SAVE_MISSING_CSV = False
SAVE_FIGURE = True
SAVE_OPTIMUM_CSV = True

OUT_DETAIL_CSV = os.path.join(OUT_DIR, "baseline_detail_merged_4db_incrementalIWUE_CORRECT.csv")
OUT_SUMMARY_CSV = os.path.join(OUT_DIR, "baseline_summary_from_detailcsv_merged_4db_incrementalIWUE_CORRECT.csv")
OUT_MISSING_CSV = os.path.join(OUT_DIR, "baseline_missing_combinations_4db_incrementalIWUE_CORRECT.csv")
OUT_FIG_PNG = os.path.join(OUT_DIR, "baseline_optimization_heatmap_from_detailcsv_merged_4db_incrementalIWUE_CORRECT.png")
OUT_OPTIMUM_CSV = os.path.join(OUT_DIR, "baseline_optimum_summary_merged_4db_incrementalIWUE_CORRECT.csv")

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 15,
})

TEXT_COLOR = "black"
NF_TEXT = "(NF)"
NF_Y_OFFSET = 0.18
NF_X_OFFSET = 0.00


# ============================================================
# HELPERS
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


def compute_optimum(stats: pd.DataFrame) -> pd.Series:
    feasible = stats[stats["Feasible"]].copy()
    if feasible.empty:
        raise ValueError("❌ Aucun traitement faisable (Y/Ymax >= seuil).")

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
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)


def draw_heatmap(
    ax,
    M: np.ndarray,
    title: str,
    cmap: str,
    cbar_label: str,
    fig,
    fmt: str,
    xlabels: list[str],
    ylabels: list[str],
    feasible_mask: np.ndarray,
    optimum_pos: tuple[int, int] | None,
    show_opt_label: bool,
) -> None:
    nrows, ncols = M.shape
    im = ax.imshow(M, cmap=cmap, aspect="equal")

    ax.set_title(title, pad=8)
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
                co + 0.28,
                ro + 0.34,
                "Opt",
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black", linewidth=0.8),
                zorder=8,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.03)
    cbar.set_label(cbar_label)


def draw_irrigation_bar(ax, stats: pd.DataFrame, irr_order: list[float]) -> None:
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
    ax.set_title("(b) Seasonal irrigation, I (mm)", pad=8)
    ax.grid(axis="y", linestyle="-", linewidth=0.4)


def export_optimum_csv(optimum: pd.Series) -> pd.DataFrame:
    optimum_df = pd.DataFrame([
        {
            "Scenario": "baseline",
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
        }
    ])

    if SAVE_OPTIMUM_CSV:
        optimum_df.to_csv(OUT_OPTIMUM_CSV, index=False, encoding="utf-8-sig")
        print(f"✅ CSV optimum enregistré : {OUT_OPTIMUM_CSV}")

    return optimum_df


# ============================================================
# EXTRACTION D'UNE DB
# ============================================================
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
    df = df[df["Scenario"] == BASELINE_SCENARIO_NAME].copy()
    if df.empty:
        print(f"⚠️ Aucune ligne baseline trouvée dans {db_path}")
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
    sow = sow[sow["SowingYear"].isin(BASELINE_YEARS)].copy()
    if sow.empty:
        print(f"⚠️ Aucun semis baseline valide dans {db_path}")
        return pd.DataFrame()

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
                matched.append({
                    "DB_Source": key[0],
                    "SimulationID": key[1],
                    "Scenario": key[2],
                    "ModelTag": key[3],
                    "Irr_pct": key[4],
                    "N_pct": key[5],
                    "SowingDate": row["SowingDate"],
                    "SowingYear": row["SowingYear"],
                    "HarvestRipeDate": pd.Timestamp(harv_dates[idx]),
                })

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
            i_season = g.loc[mask, "Irr"].sum()

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

            seasonal_rows.append({
                "DB_Source": key[0],
                "SimulationID": key[1],
                "Scenario": key[2],
                "ModelTag": key[3],
                "Irr_pct": float(key[4]),
                "N_pct": float(key[5]),
                "SowingYear": int(wr["SowingYear"]),
                "SowingDate": sd,
                "HarvestRipeDate": hd,
                "SeasonalIrrigation": float(i_season),
                "Y_final": float(pd.to_numeric(g_h["Y_final_raw"].iloc[0], errors="coerce")),
                "Nres_final": float(pd.to_numeric(g_h["Nres_kg_ha"].iloc[0], errors="coerce")),
            })

    seasonal = pd.DataFrame(seasonal_rows)
    if seasonal.empty:
        print(f"⚠️ Aucune métrique saisonnière calculée dans {db_path}")
        return pd.DataFrame()

    seasonal["Treatment"] = seasonal.apply(lambda r: f"{r['Irr_pct']:g}%ETc×{r['N_pct']:g}%N", axis=1)
    seasonal["TreatmentLabel"] = seasonal.apply(lambda r: f"{r['Irr_pct']:g}% ETc × {r['N_pct']:g}% N", axis=1)
    return seasonal


# ============================================================
# ETAPE 1 : CREER LE CSV DETAILLE
# ============================================================
def build_detail_dataframe_from_dbs() -> pd.DataFrame:
    all_seasonal = []
    for db_path in DB_PATHS:
        seasonal_db = extract_one_db(db_path)
        if not seasonal_db.empty:
            all_seasonal.append(seasonal_db)

    if not all_seasonal:
        raise ValueError("❌ Aucune donnée valide extraite depuis les DB.")

    seasonal = pd.concat(all_seasonal, ignore_index=True)

    seasonal = seasonal.drop_duplicates(
        subset=["DB_Source", "ModelTag", "Irr_pct", "N_pct", "SowingYear", "SowingDate", "HarvestRipeDate"]
    ).copy()

    rainfed = seasonal.loc[seasonal["Irr_pct"] == 0, ["ModelTag", "SowingYear", "N_pct", "Y_final"]].copy()
    rainfed = rainfed.groupby(["ModelTag", "SowingYear", "N_pct"], as_index=False).agg(Y_rainfed=("Y_final", "mean"))
    seasonal = seasonal.merge(rainfed, on=["ModelTag", "SowingYear", "N_pct"], how="left")

    seasonal["DeltaY"] = (seasonal["Y_final"] - seasonal["Y_rainfed"]).clip(lower=0)
    seasonal["IWUE_tha_mm"] = np.where(
        seasonal["SeasonalIrrigation"] > 0,
        seasonal["DeltaY"] / seasonal["SeasonalIrrigation"],
        np.nan,
    )
    seasonal["IWUE_kg_m3"] = seasonal["IWUE_tha_mm"] * 100.0
    seasonal["IsRainfed"] = seasonal["Irr_pct"].eq(0)
    seasonal["ScenarioLabel"] = seasonal["Scenario"]

    detail_cols = [
        "DB_Source",
        "SimulationID",
        "Scenario",
        "ScenarioLabel",
        "ModelTag",
        "Treatment",
        "TreatmentLabel",
        "Irr_pct",
        "N_pct",
        "SowingYear",
        "SowingDate",
        "HarvestRipeDate",
        "SeasonalIrrigation",
        "Y_final",
        "Y_rainfed",
        "DeltaY",
        "IWUE_tha_mm",
        "IWUE_kg_m3",
        "Nres_final",
        "IsRainfed",
    ]

    detail_df = seasonal[detail_cols].sort_values(
        ["DB_Source", "ModelTag", "Irr_pct", "N_pct", "SowingYear"],
        ascending=[True, True, False, False, True],
    ).reset_index(drop=True)

    if SAVE_DETAIL_CSV:
        detail_df.to_csv(OUT_DETAIL_CSV, index=False, encoding="utf-8-sig")
        print(f"✅ CSV détaillé enregistré : {OUT_DETAIL_CSV}")

    return detail_df


# ============================================================
# ETAPE 2 : CALCULER LE RESUME A PARTIR DU CSV DETAILLE
# ============================================================
def build_stats_from_detail_df(detail_df: pd.DataFrame) -> pd.DataFrame:
    df = detail_df.copy()

    numeric_cols = [
        "Irr_pct", "N_pct", "Y_final", "Y_rainfed", "DeltaY",
        "SeasonalIrrigation", "IWUE_tha_mm", "IWUE_kg_m3", "Nres_final"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    by_model = df.groupby(["ModelTag", "Irr_pct", "N_pct"], as_index=False).agg(
        Y_model_mean=("Y_final", "mean"),
        Yrf_model_mean=("Y_rainfed", "mean"),
        DeltaY_model_mean=("DeltaY", "mean"),
        I_model_mean=("SeasonalIrrigation", "mean"),
        IWUE_model_mean=("IWUE_kg_m3", "mean"),
        Nres_model_mean=("Nres_final", "mean"),
        n_years=("SowingYear", "nunique"),
        n_obs=("SowingYear", "size"),
    )

    stats = by_model.groupby(["Irr_pct", "N_pct"], as_index=False).agg(
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
        n_models_total=("ModelTag", "count"),
        n_models_unique=("ModelTag", "nunique"),
        mean_years_per_model=("n_years", "mean"),
        total_obs=("n_obs", "sum"),
    )

    for c in ["Y_std", "I_std", "IWUE_std_kgm3", "Nres_std"]:
        stats[c] = stats[c].fillna(0.0)

    nmax = stats.groupby("Irr_pct", as_index=False)["N_pct"].max().rename(columns={"N_pct": "N_max"})
    stats = stats.merge(nmax, on="Irr_pct", how="left")

    y_nmax = (
        stats.loc[stats["N_pct"] == stats["N_max"], ["Irr_pct", "Y_mean"]]
        .drop_duplicates(subset=["Irr_pct"])
        .rename(columns={"Y_mean": "Y_Nmax_sameIrr"})
    )
    stats = stats.merge(y_nmax, on="Irr_pct", how="left")

    if stats["Y_Nmax_sameIrr"].isna().any():
        stats["Y_Nmax_sameIrr"] = stats["Y_Nmax_sameIrr"].fillna(
            stats.groupby("Irr_pct")["Y_mean"].transform("max")
        )

    stats["Yield_ratio_to_Nmax"] = stats["Y_mean"] / stats["Y_Nmax_sameIrr"].replace(0, np.nan)
    stats["Feasible"] = (stats["Yield_ratio_to_Nmax"] >= YIELD_SUFFICIENCY_THRESHOLD).fillna(False).astype(bool)
    stats["TreatmentLabel"] = stats.apply(lambda r: f"{r['Irr_pct']:g}% ETc × {r['N_pct']:g}% N", axis=1)

    expected_levels = list(range(0, 101, 5))
    expected = pd.DataFrame([(i, n) for i in expected_levels for n in expected_levels], columns=["Irr_pct", "N_pct"])
    present = stats[["Irr_pct", "N_pct"]].drop_duplicates().copy()
    missing = expected.merge(present, on=["Irr_pct", "N_pct"], how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns="_merge")

    if SAVE_MISSING_CSV:
        missing.to_csv(OUT_MISSING_CSV, index=False, encoding="utf-8-sig")
        print(f"✅ CSV des combinaisons manquantes enregistré : {OUT_MISSING_CSV}")

    print(f"✅ Nombre de combinaisons présentes : {len(present)} / 441")
    print(f"✅ Nombre de combinaisons manquantes : {len(missing)}")

    stats = stats.sort_values(["Irr_pct", "N_pct"], ascending=[False, False]).reset_index(drop=True)

    if SAVE_SUMMARY_CSV:
        stats.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
        print(f"✅ CSV résumé enregistré : {OUT_SUMMARY_CSV}")

    return stats


def build_stats_from_detail_csv(detail_csv_path: str) -> pd.DataFrame:
    if not os.path.exists(detail_csv_path):
        raise FileNotFoundError(f"❌ CSV détaillé introuvable : {detail_csv_path}")

    detail_df = pd.read_csv(detail_csv_path, encoding="utf-8-sig")
    return build_stats_from_detail_df(detail_df)


# ============================================================
# ETAPE 3 : EXPORT OPTIMUM DEPUIS LE RESUME
# ============================================================
def export_optimum_from_stats(stats: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    optimum = compute_optimum(stats)
    optimum_df = export_optimum_csv(optimum)

    print("\n==============================")
    print("Optimum (calculé depuis le CSV détaillé)")
    print("==============================")
    print(f"Traitement optimal : {optimum['Irr_pct']:g}% ETc × {optimum['N_pct']:g}% N")
    print(f"Y_mean      = {optimum['Y_mean']:.3f}")
    print(f"Y_rainfed   = {optimum['Y_rainfed_mean']:.3f}")
    print(f"DeltaY_mean = {optimum['DeltaY_mean']:.3f}")
    print(f"I_mean      = {optimum['I_mean']:.1f}")
    print(f"IWUE        = {optimum['IWUE_mean_kgm3']:.2f}")
    print(f"Nres        = {optimum['Nres_mean']:.2f}")
    print(f"Y/Ymax      = {optimum['Yield_ratio_to_Nmax']:.3f}")
    print(f"Feasible    = {bool(optimum['Feasible'])}")

    return optimum, optimum_df


# ============================================================
# PLOT
# ============================================================
def baseline_plot_main(stats: pd.DataFrame) -> None:
    numeric_cols = [
        "Irr_pct", "N_pct", "Y_mean", "Y_rainfed_mean", "DeltaY_mean",
        "I_mean", "IWUE_mean_kgm3", "Nres_mean", "Yield_ratio_to_Nmax",
    ]
    for c in numeric_cols:
        if c in stats.columns:
            stats[c] = pd.to_numeric(stats[c], errors="coerce")

    if "Feasible" not in stats.columns:
        raise ValueError("❌ Colonne 'Feasible' absente du DataFrame résumé.")

    stats = stats.dropna(subset=["Irr_pct", "N_pct"]).copy()
    if stats.empty:
        raise ValueError("❌ Le DataFrame résumé est vide ou invalide.")

    optimum = compute_optimum(stats)
    irr_order = sorted(stats["Irr_pct"].unique().tolist(), reverse=True)
    n_order = sorted(stats["N_pct"].unique().tolist(), reverse=True)

    M_Y = matrix_from_stats(stats, "Y_mean", irr_order, n_order)
    M_IWUE = matrix_from_stats(stats, "IWUE_mean_kgm3", irr_order, n_order)
    M_Nres = matrix_from_stats(stats, "Nres_mean", irr_order, n_order)
    M_Feasible = matrix_bool_from_stats(stats, "Feasible", irr_order, n_order)

    r_opt = irr_order.index(float(optimum["Irr_pct"]))
    c_opt = n_order.index(float(optimum["N_pct"]))
    opt_pos = (r_opt, c_opt)

    xlabels = [f"{v:g}% N" for v in n_order]
    ylabels = [f"{v:g}% ETc" for v in irr_order]

    fig_w = min(24, 12.6 + max(0, (len(n_order) - 3)) * 0.6)
    fig_h = min(24, 10.4 + max(0, (len(irr_order) - 3)) * 0.6)

    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), constrained_layout=True)

    draw_heatmap(
        axes[0, 0], M_Y,
        "(a) Final biomass at harvest, Y (t ha$^{-1}$)",
        "YlGn", "t ha$^{-1}$", fig, "{:.2f}",
        xlabels, ylabels, M_Feasible, opt_pos, False
    )

    draw_irrigation_bar(axes[0, 1], stats, irr_order)

    draw_heatmap(
        axes[1, 0], M_IWUE,
        "(c) Incremental irrigation water use efficiency, IWUE (kg m$^{-3}$)",
        "PuBuGn", "kg m$^{-3}$", fig, "{:.2f}",
        xlabels, ylabels, M_Feasible, opt_pos, True
    )

    draw_heatmap(
        axes[1, 1], M_Nres,
        "(d) Residual mineral N at harvest (kg N ha$^{-1}$)",
        "YlOrRd", "kg N ha$^{-1}$", fig, "{:.1f}",
        xlabels, ylabels, M_Feasible, opt_pos, False
    )

    fig.suptitle("Baseline irrigation–nitrogen optimization", y=1.02)
    footer = (
        f"IWUE = (Y_i - Y_rainfed) / I ; Y_rainfed matched by model, year, and N level across all merged DB. "
        f"Optimum: max IWUE among treatments meeting Y/Ymax (same ETc) ≥ {YIELD_SUFFICIENCY_THRESHOLD:.2f}; "
        f"ties within {IWUE_COMPARABLE_TOL_REL*100:.0f}% resolved by minimum residual N. "
        f"(NF) = non-feasible."
    )
    fig.text(0.5, -0.01, footer, ha="center", va="top", fontsize=9)

    if SAVE_FIGURE:
        plt.savefig(OUT_FIG_PNG, dpi=400, bbox_inches="tight")
        print(f"✅ Figure enregistrée : {OUT_FIG_PNG}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # 1) Construire le CSV détaillé depuis les DB
    detail_df = build_detail_dataframe_from_dbs()

    # 2) Recharger le CSV détaillé et recalculer le résumé depuis ce CSV
    stats_df = build_stats_from_detail_csv(OUT_DETAIL_CSV)

    # 3) Exporter l'optimum
    optimum, optimum_df = export_optimum_from_stats(stats_df)

    # 4) Tracer la figure à partir du résumé dérivé du CSV détaillé
    baseline_plot_main(stats_df)

    print("\nPetit tableau optimum :")
    print(optimum_df.to_string(index=False))