"""
visualizer.py
=============
Responsabilidad única: generar y guardar todos los gráficos del proyecto.

Gráficos producidos
-------------------
1. plot_r16_probabilities  → barras horizontales por partido (prob A vs B)
2. plot_montecarlo_heatmap → heatmap de % por ronda para todas las selecciones
3. plot_winner_odds        → barras verticales con % de llegar a la Final
4. plot_elo_ranking        → ranking Elo de las selecciones clasificadas

Todos los gráficos se guardan en outputs/ con 150 dpi.
Soporta cualquier conjunto de rondas activas (pipeline iterativo).
"""

import logging
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Logger ────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ── Rutas ─────────────────────────────────────────────────────────────────────

OUTPUTS_DIR = Path(__file__).parents[1] / "outputs"

# ── Paleta de colores ─────────────────────────────────────────────────────────

C_PRIMARY   = "#1a1a2e"
C_SECONDARY = "#16213e"
C_ACCENT    = "#0f3460"
C_HIGHLIGHT = "#e94560"
C_GOLD      = "#f5a623"
C_TEXT      = "#eaeaea"
C_GRID      = "#2e2e4e"

CONFIDENCE_COLORS = {
    "Alta":  "#27ae60",
    "Media": "#f39c12",
    "Baja":  "#e74c3c",
}

# Etiquetas legibles por nombre de ronda interno
ROUND_LABELS = {
    "Round of 32":    "16avos",
    "Round of 16":    "8avos",
    "Quarter-finals": "Cuartos",
    "Semi-finals":    "Semis",
    "Final":          "Final",
    "Winner":         "Campeón",
}


# ── Funciones públicas ────────────────────────────────────────────────────────

def plot_r16_probabilities(df_pred: pd.DataFrame, round_label: str = "16avos de final") -> Path:
    """
    Barras horizontales apiladas para los cruces de una ronda.
    Cada barra muestra P(A) y P(B). El color del texto indica confianza.

    Parameters
    ----------
    df_pred : pd.DataFrame
        Salida de ``match_predictor.predict_bracket()``.
    round_label : str
        Nombre de la ronda para el título del gráfico.

    Returns
    -------
    Path al fichero PNG guardado.
    """
    df = df_pred.sort_values("prob_a", ascending=True).reset_index(drop=True)
    n  = len(df)

    fig, ax = plt.subplots(figsize=(14, max(6, n * 0.65)))
    fig.patch.set_facecolor(C_PRIMARY)
    ax.set_facecolor(C_SECONDARY)

    for i, row in df.iterrows():
        ax.barh(i, row["prob_a"], color=C_ACCENT,    height=0.6, left=0)
        ax.barh(i, row["prob_b"], color=C_HIGHLIGHT, height=0.6, left=row["prob_a"])

        ax.text(0.01, i, f"  {row['team_a']}  {row['prob_a']*100:.1f}%",
                va="center", ha="left", color=C_TEXT, fontsize=9, fontweight="bold")
        ax.text(0.99, i, f"{row['prob_b']*100:.1f}%  {row['team_b']}  ",
                va="center", ha="right", color=C_TEXT, fontsize=9, fontweight="bold")

        conf_color = CONFIDENCE_COLORS.get(row.get("confidence", "Baja"), C_TEXT)
        ax.axhline(i, color=conf_color, linewidth=0.4, alpha=0.3)

    ax.axvline(0.5, color=C_TEXT, linewidth=1, linestyle="--", alpha=0.5)
    ax.text(0.5, n - 0.1, "50%", ha="center", color=C_TEXT, fontsize=8, alpha=0.6)

    patches = [
        mpatches.Patch(color=v, label=f"Confianza {k}")
        for k, v in CONFIDENCE_COLORS.items()
    ]
    ax.legend(handles=patches, loc="lower right",
              facecolor=C_ACCENT, edgecolor=C_GRID, labelcolor=C_TEXT, fontsize=8)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.tick_params(colors=C_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_GRID)

    ax.set_title(f"CupCast26 · Predicciones {round_label}",
                 color=C_TEXT, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Probabilidad de clasificación", color=C_TEXT, fontsize=10)

    plt.tight_layout()
    safe_label = round_label.replace(" ", "_").replace("/", "_")
    return _save(fig, f"probabilities_{safe_label}.png")


def plot_montecarlo_heatmap(df_mc: pd.DataFrame) -> Path:
    """
    Heatmap con el % de simulaciones en que cada selección alcanzó cada ronda.
    Muestra solo las rondas presentes en df_mc (soporta pipeline iterativo).

    Parameters
    ----------
    df_mc : pd.DataFrame
        Salida de ``montecarlo.run_montecarlo()``.

    Returns
    -------
    Path al fichero PNG guardado.
    """
    # Detectar rondas presentes (columnas que no son 'team' ni 'elo')
    round_cols   = [c for c in df_mc.columns if c not in ("team", "elo")]
    round_labels = [ROUND_LABELS.get(c, c) for c in round_cols]

    df     = df_mc.sort_values("Final", ascending=False).reset_index(drop=True)
    matrix = df[round_cols].values

    fig, ax = plt.subplots(figsize=(max(8, len(round_cols) * 1.6), max(10, len(df) * 0.5)))
    fig.patch.set_facecolor(C_PRIMARY)
    ax.set_facecolor(C_PRIMARY)

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)

    for i in range(len(df)):
        for j in range(len(round_cols)):
            val   = matrix[i, j]
            color = "black" if val > 55 else C_TEXT
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(range(len(round_cols)))
    ax.set_xticklabels(round_labels, color=C_TEXT, fontsize=10)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["team"], color=C_TEXT, fontsize=9)
    ax.tick_params(colors=C_TEXT, length=0)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.025)
    cbar.ax.tick_params(colors=C_TEXT)
    cbar.set_label("% simulaciones", color=C_TEXT, fontsize=9)

    ax.set_title("CupCast26 · Probabilidad por ronda — 50 000 simulaciones",
                 color=C_TEXT, fontsize=13, fontweight="bold", pad=15)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_GRID)

    plt.tight_layout()
    return _save(fig, "montecarlo_heatmap.png")


def plot_winner_odds(df_mc: pd.DataFrame) -> Path:
    """
    Barras verticales con el % de veces que cada selección llegó a la Final.
    Solo muestra equipos con Final > 0%.

    Parameters
    ----------
    df_mc : pd.DataFrame
        Salida de ``montecarlo.run_montecarlo()``.

    Returns
    -------
    Path al fichero PNG guardado.
    """
    df = (
        df_mc[df_mc["Final"] > 0]
        .sort_values("Final", ascending=False)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(C_PRIMARY)
    ax.set_facecolor(C_SECONDARY)

    colors = [C_GOLD if i == 0 else C_ACCENT for i in range(len(df))]
    bars   = ax.bar(df["team"], df["Final"], color=colors,
                    edgecolor=C_GRID, linewidth=0.5)

    for bar, val in zip(bars, df["Final"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}%",
                ha="center", va="bottom",
                color=C_TEXT, fontsize=8, fontweight="bold")

    ax.set_ylim(0, df["Final"].max() * 1.2)
    ax.set_ylabel("% de simulaciones llegando a la Final", color=C_TEXT, fontsize=10)
    ax.set_title("CupCast26 · Probabilidad de llegar a la Final",
                 color=C_TEXT, fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(axis="x", colors=C_TEXT, rotation=35, labelsize=9)
    ax.tick_params(axis="y", colors=C_TEXT)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis="y", color=C_GRID, linewidth=0.5, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_GRID)

    plt.tight_layout()
    return _save(fig, "winner_odds.png")


def plot_elo_ranking(df_elo: pd.DataFrame) -> Path:
    """
    Barras horizontales con el rating Elo de las selecciones,
    ordenadas de mayor a menor.

    Parameters
    ----------
    df_elo : pd.DataFrame
        Salida de ``elo_model.elo_summary()``.

    Returns
    -------
    Path al fichero PNG guardado.
    """
    df = df_elo.sort_values("elo", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(8, len(df) * 0.4)))
    fig.patch.set_facecolor(C_PRIMARY)
    ax.set_facecolor(C_SECONDARY)

    norm   = plt.Normalize(df["elo"].min(), df["elo"].max())
    colors = plt.cm.YlOrRd(norm(df["elo"].values))

    bars = ax.barh(df["team"], df["elo"], color=colors,
                   edgecolor=C_GRID, linewidth=0.4, height=0.7)

    for bar, val in zip(bars, df["elo"]):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", color=C_TEXT, fontsize=8)

    ax.set_xlim(df["elo"].min() - 50, df["elo"].max() + 80)
    ax.set_xlabel("Rating Elo (histórico ponderado)", color=C_TEXT, fontsize=10)
    ax.set_title("CupCast26 · Rating Elo — selecciones clasificadas",
                 color=C_TEXT, fontsize=13, fontweight="bold", pad=15)
    ax.tick_params(axis="x", colors=C_TEXT)
    ax.tick_params(axis="y", colors=C_TEXT, labelsize=9)
    ax.grid(axis="x", color=C_GRID, linewidth=0.5, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_GRID)

    plt.tight_layout()
    return _save(fig, "elo_ranking.png")


# ── Función privada ───────────────────────────────────────────────────────────

def _save(fig: plt.Figure, filename: str) -> Path:
    """Guarda la figura en outputs/ y libera memoria."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("    Gráfico guardado: %s", path)
    return path