"""
accuracy.py
===========
Responsabilidad única: comparar las predicciones del modelo contra
los resultados reales de cada partido del Mundial 2026.

Lógica
------
El CSV de martj42 contiene los partidos ya jugados del Mundial con
su resultado real. Para cada partido predicho, se compara:

    - Si el modelo predijo al ganador correcto → acierto
    - Margen de error: |prob_predicha - 1.0| si acertó, |prob_predicha - 0.0| si falló

El accuracy se acumula por ronda para ver si el modelo mejora
conforme avanza el torneo (el Elo se actualiza con resultados reales).
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

# Nombre del torneo tal como aparece en el CSV de martj42
WC_2026_TOURNAMENT = "FIFA World Cup"
WC_2026_START      = pd.Timestamp("2026-06-11")

# Mapeo ronda FIFA → nombre interno
ROUND_MAP = {
    "Round of 32":    "16avos",
    "Round of 16":    "8avos",
    "Quarter-finals": "Cuartos",
    "Semi-finals":    "Semis",
    "Final":          "Final",
}


@dataclass
class MatchResult:
    """Resultado real vs predicción de un partido concreto."""
    team_a:       str
    team_b:       str
    round_name:   str
    prob_a:       float          # probabilidad predicha para team_a
    prob_b:       float
    predicted_winner: str
    real_winner:  str
    correct:      bool
    confidence:   str
    date:         str


@dataclass
class AccuracyReport:
    """Informe de precisión del modelo por ronda."""
    total_matches:   int = 0
    correct:         int = 0
    by_round:        dict = field(default_factory=dict)
    match_results:   list = field(default_factory=list)

    @property
    def overall_accuracy(self) -> float:
        if self.total_matches == 0:
            return 0.0
        return round(self.correct / self.total_matches * 100, 2)


def extract_wc2026_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae los partidos del Mundial 2026 ya jugados del CSV histórico.

    Parameters
    ----------
    df : pd.DataFrame
        Salida de ``data_loader.load_raw_data()``.

    Returns
    -------
    pd.DataFrame con los partidos del Mundial 2026 con resultado real.
    """
    mask = (
        (df["tournament"] == WC_2026_TOURNAMENT) &
        (df["date"] >= WC_2026_START) &
        (df["home_score"].notna()) &
        (df["away_score"].notna())
    )
    wc = df[mask].copy()
    logger.info("Partidos del Mundial 2026 con resultado real: %d", len(wc))
    return wc.reset_index(drop=True)


def compute_accuracy(
    wc_results: pd.DataFrame,
    predictions: dict[str, dict],
) -> AccuracyReport:
    """
    Compara los resultados reales con las predicciones almacenadas.

    Parameters
    ----------
    wc_results : pd.DataFrame
        Salida de ``extract_wc2026_results()``.
    predictions : dict[str, dict]
        Caché de predicciones: {(team_a, team_b): resultado de predict_match}.
        La clave puede ser en cualquier orden — se prueba ambas direcciones.

    Returns
    -------
    AccuracyReport con precisión global y por ronda.
    """
    report = AccuracyReport()

    for _, row in wc_results.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        # Buscar predicción en caché (en cualquier dirección)
        pred = predictions.get((home, away)) or predictions.get((away, home))
        if pred is None:
            logger.debug("Sin predicción para %s vs %s — omitiendo.", home, away)
            continue

        # Ganador real
        if row["home_score"] > row["away_score"]:
            real_winner = home
        elif row["home_score"] < row["away_score"]:
            real_winner = away
        else:
            # Empate en fase de grupos — no aplica en eliminatorias
            real_winner = "Draw"

        predicted_winner = pred["favorite"]
        correct          = (predicted_winner == real_winner)

        round_name = _infer_round(row["date"], wc_results)

        match_result = MatchResult(
            team_a           = pred["team_a"],
            team_b           = pred["team_b"],
            round_name       = round_name,
            prob_a           = pred["prob_a"],
            prob_b           = pred["prob_b"],
            predicted_winner = predicted_winner,
            real_winner      = real_winner,
            correct          = correct,
            confidence       = pred.get("confidence", "—"),
            date             = str(row["date"].date()),
        )

        report.total_matches += 1
        if correct:
            report.correct += 1

        if round_name not in report.by_round:
            report.by_round[round_name] = {"total": 0, "correct": 0}
        report.by_round[round_name]["total"]   += 1
        report.by_round[round_name]["correct"] += int(correct)
        report.match_results.append(match_result)

    # Calcular accuracy por ronda
    for rnd, data in report.by_round.items():
        data["accuracy"] = round(
            data["correct"] / data["total"] * 100, 2
        ) if data["total"] > 0 else 0.0

    logger.info(
        "Accuracy global: %d/%d (%.1f%%)",
        report.correct, report.total_matches, report.overall_accuracy,
    )
    return report


def _infer_round(date: pd.Timestamp, wc_results: pd.DataFrame) -> str:
    """
    Infiere la ronda de un partido según su fecha relativa al torneo.
    Aproximación por bloques de fechas del Mundial 2026.
    """
    days = (date - WC_2026_START).days
    if days < 16:
        return "Fase de grupos"
    if days < 22:
        return "16avos"
    if days < 26:
        return "8avos"
    if days < 29:
        return "Cuartos"
    if days < 32:
        return "Semis"
    return "Final"