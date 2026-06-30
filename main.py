"""
main.py
=======
Punto de entrada del proyecto CupCast26.

Pipeline iterativo por rondas
------------------------------
El modelo se actualiza ronda a ronda conforme avanzan los resultados reales.
Para avanzar de ronda:

    1. Rellena el bracket de la siguiente ronda en BRACKET_ACTUAL
       con los ganadores reales de la ronda anterior.
    2. Ajusta RONDA_ACTUAL al nombre de la nueva ronda.
    3. Ejecuta: python main.py

El Elo se recalcula automáticamente incluyendo los partidos jugados
en este Mundial (ya están en el CSV de martj42, que se actualiza en tiempo real).
Si el CSV ya tiene los resultados de 16avos, el Elo los habrá absorbido
con peso máximo (combined_weight cercano a 1.5 × 1.0).

Rondas válidas para RONDA_ACTUAL
---------------------------------
    "Round of 32"     → 16avos  (bracket con 16 cruces)
    "Round of 16"     → 8avos   (bracket con  8 cruces)
    "Quarter-finals"  → Cuartos (bracket con  4 cruces)
    "Semi-finals"     → Semis   (bracket con  2 cruces)
    "Final"           → Final   (bracket con  1 cruce)

Uso
---
    python main.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader         import load_raw_data
from src.feature_engineering import engineer_features, save_processed
from src.elo_model           import compute_elo_ratings, elo_summary
from src.match_predictor     import predict_bracket
from src.montecarlo          import run_montecarlo, ROUND_LABELS
from src.visualizer          import (
    plot_r16_probabilities,
    plot_montecarlo_heatmap,
    plot_winner_odds,
    plot_elo_ranking,
)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s · %(message)s")
logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN — actualizar entre rondas
# ═════════════════════════════════════════════════════════════════════════════

# Ronda que se va a predecir
RONDA_ACTUAL: str = "Round of 32"   # cambiar a "Round of 16", "Quarter-finals"…

# Bracket de la ronda actual con los cruces reales
# Al pasar de ronda: sustituir por los ganadores reales de la ronda anterior
BRACKET_ACTUAL: list[tuple[str, str]] = [
    # ── 16avos de final — Mundial 2026 ──────────────────────────────────────
    ("South Africa",         "Canada"),
    ("Brazil",               "Japan"),
    ("Germany",              "Paraguay"),
    ("Netherlands",          "Morocco"),
    ("Ivory Coast",          "Norway"),
    ("France",               "Sweden"),
    ("Mexico",               "Ecuador"),
    ("England",              "DR Congo"),
    ("Belgium",              "Senegal"),
    ("United States",        "Bosnia and Herzegovina"),
    ("Spain",                "Austria"),
    ("Portugal",             "Croatia"),
    ("Switzerland",          "Algeria"),
    ("Australia",            "Egypt"),
    ("Argentina",            "Cape Verde"),
    ("Colombia",             "Ghana"),
    # ── Ejemplo de cómo quedaría en 8avos (descomentar cuando proceda): ──
    # ("Brazil",      "Germany"),
    # ("Netherlands", "France"),
    # ("England",     "Belgium"),
    # ("Spain",       "Portugal"),
    # ("Argentina",   "Colombia"),
    # ("Switzerland", "Australia"),
    # ("South Africa","Mexico"),
    # ("Ivory Coast", "United States"),
]

N_SIMULATIONS: int = 50_000

# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:

    round_label = ROUND_LABELS.get(RONDA_ACTUAL, RONDA_ACTUAL)
    logger.info("CupCast26 · Predicción desde: %s", round_label)

    # ── 1. Datos ──────────────────────────────────────────────────────────────
    logger.info("═══ 1 · Cargando datos crudos ═══")
    raw = load_raw_data()
    logger.info("    %d partidos · %s → %s",
                len(raw), raw["date"].min().date(), raw["date"].max().date())

    # ── 2. Feature engineering ────────────────────────────────────────────────
    logger.info("═══ 2 · Feature engineering ═══")
    processed = engineer_features(raw)
    save_processed(processed)

    # ── 3. Modelo Elo ─────────────────────────────────────────────────────────
    logger.info("═══ 3 · Calculando ratings Elo ═══")
    ratings   = compute_elo_ratings(processed)
    all_teams = list({t for pair in BRACKET_ACTUAL for t in pair})
    df_elo    = elo_summary(ratings, all_teams)
    logger.info("\n%s", df_elo.to_string(index=False))

    # ── 4. Predicción por partido ─────────────────────────────────────────────
    logger.info("═══ 4 · Predicciones %s ═══", round_label)
    df_pred = predict_bracket(BRACKET_ACTUAL, ratings, processed)
    cols    = ["team_a", "team_b", "prob_a", "prob_b", "favorite", "confidence"]
    logger.info("\n%s", df_pred[cols].to_string(index=False))

    # ── 5. Montecarlo ─────────────────────────────────────────────────────────
    logger.info("═══ 5 · Simulación Montecarlo (%d iteraciones) ═══", N_SIMULATIONS)
    df_mc = run_montecarlo(
        bracket        = BRACKET_ACTUAL,
        ratings        = ratings,
        df_processed   = processed,
        n_simulations  = N_SIMULATIONS,
        seed           = 42,
        starting_round = RONDA_ACTUAL,
    )
    logger.info("\n%s", df_mc.to_string(index=False))

    # ── 6. Visualización ──────────────────────────────────────────────────────
    logger.info("═══ 6 · Generando visualizaciones ═══")
    Path("outputs").mkdir(exist_ok=True)

    plot_r16_probabilities(df_pred, round_label=round_label)
    plot_montecarlo_heatmap(df_mc)
    plot_winner_odds(df_mc)
    plot_elo_ranking(df_elo)

    logger.info("═══ Pipeline completado · gráficos en outputs/ ═══")


if __name__ == "__main__":
    main()