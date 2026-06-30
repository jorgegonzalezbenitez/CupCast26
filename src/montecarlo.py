"""
montecarlo.py
=============
Responsabilidad única: simular el bracket de eliminación directa
mediante el método de Montecarlo.

Rondas del Mundial 2026 (eliminación directa)
----------------------------------------------
    Round of 32   → 16avos   (16 partidos, 32 equipos)
    Round of 16   → 8avos    (8 partidos,  16 equipos)
    Quarter-finals → Cuartos  (4 partidos,  8 equipos)
    Semi-finals   → Semis    (2 partidos,  4 equipos)
    Final         → Final    (1 partido,   2 equipos)
    Winner        → Campeón

Idea central
------------
Se simulan N torneos completos desde la ronda actual hacia adelante.
En cada simulación, cada partido se resuelve tirando un número aleatorio
contra la probabilidad del predictor: si rand() < P(A), avanza A.

Pipeline iterativo
------------------
El modelo soporta arrancar desde cualquier ronda:
    - Si estamos en 16avos, el bracket tiene 16 cruces
    - Si estamos en 8avos, el bracket tiene 8 cruces (ganadores reales de 16avos)
    - Y así sucesivamente

Lógica de acumulación
---------------------
Por cada simulación, cada ganador de cada ronda se registra en reach_count.
El campeón se registra adicionalmente en "Winner".
Las sumas por columna verifican la corrección:
    Round of 32     → suma 1600%  (32 equipos × 50%)
    Round of 16     → suma  800%
    Quarter-finals  → suma  400%
    Semi-finals     → suma  200%
    Final           → suma  100%
    Winner          → suma  100%
"""

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.match_predictor import predict_match

# ── Logger ────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ── Rondas del Mundial 2026 en orden cronológico ──────────────────────────────

ALL_ROUNDS = [
    "Round of 32",      # 16avos
    "Round of 16",      # 8avos
    "Quarter-finals",   # Cuartos
    "Semi-finals",      # Semis
    "Final",
    "Winner",
]

ROUND_LABELS = {
    "Round of 32":    "16avos",
    "Round of 16":    "8avos",
    "Quarter-finals": "Cuartos",
    "Semi-finals":    "Semis",
    "Final":          "Final",
    "Winner":         "Campeón",
}


# ── Función pública principal ─────────────────────────────────────────────────

def run_montecarlo(
    bracket: list[tuple[str, str]],
    ratings: dict[str, float],
    df_processed: pd.DataFrame,
    n_simulations: int = 50_000,
    seed: int = 42,
    starting_round: str = "Round of 32",
) -> pd.DataFrame:
    """
    Simula el torneo completo N veces desde la ronda indicada.

    Soporta arrancar desde cualquier ronda, lo que permite actualizar
    las predicciones conforme avanzan los resultados reales:

        - 16avos en juego  → starting_round="Round of 32",    bracket con 16 cruces
        - 8avos  en juego  → starting_round="Round of 16",    bracket con  8 cruces
        - Cuartos en juego → starting_round="Quarter-finals", bracket con  4 cruces
        - Semis  en juego  → starting_round="Semi-finals",    bracket con  2 cruces
        - Final  en juego  → starting_round="Final",          bracket con  1 cruce

    Parameters
    ----------
    bracket : list[tuple[str, str]]
        Cruces de la ronda actual. El orden determina los emparejamientos
        de rondas posteriores: w[0] vs w[1], w[2] vs w[3], etc.
    ratings : dict[str, float]
        Salida de ``elo_model.compute_elo_ratings()``.
        Si se han jugado partidos nuevos desde el último cálculo,
        recalcular ratings antes de llamar a esta función.
    df_processed : pd.DataFrame
        Salida de ``feature_engineering.engineer_features()``.
    n_simulations : int
        Número de torneos simulados. 50 000 ofrece buen equilibrio.
    seed : int
        Semilla para reproducibilidad.
    starting_round : str
        Nombre de la ronda desde la que se simula. Debe ser una de ALL_ROUNDS.

    Returns
    -------
    pd.DataFrame con columnas [team, elo] más una columna por ronda
    simulada, con el % de simulaciones en que la selección superó esa ronda.
    Ordenado por la columna "Final" descendente.
    """
    _validate_bracket(bracket, starting_round)

    rng         = np.random.default_rng(seed)
    prob_cache: dict[tuple[str, str], float] = {}
    reach_count: dict[str, dict[str, int]]   = defaultdict(lambda: defaultdict(int))

    # Rondas que se van a simular (desde starting_round hasta Winner)
    start_idx    = ALL_ROUNDS.index(starting_round)
    active_rounds = ALL_ROUNDS[start_idx:]   # incluye "Winner" al final

    logger.info(
        "Iniciando %d simulaciones desde '%s' con %d cruces …",
        n_simulations, starting_round, len(bracket),
    )

    for sim_idx in range(n_simulations):
        _simulate_one(
            bracket, active_rounds, ratings, df_processed,
            rng, prob_cache, reach_count,
        )
        if (sim_idx + 1) % 10_000 == 0:
            logger.debug("  %d / %d simulaciones completadas.", sim_idx + 1, n_simulations)

    logger.info("Simulaciones completadas.")
    return _build_dataframe(bracket, ratings, reach_count, n_simulations, active_rounds)


def get_cached_prob(
    team_a: str,
    team_b: str,
    ratings: dict[str, float],
    df_processed: pd.DataFrame,
    prob_cache: dict[tuple[str, str], float],
) -> float:
    """Devuelve P(A gana) usando caché para evitar recalcular en cada simulación."""
    key = (team_a, team_b)
    if key not in prob_cache:
        pred = predict_match(team_a, team_b, ratings, df_processed)
        prob_cache[(team_a, team_b)] = pred["prob_a"]
        prob_cache[(team_b, team_a)] = pred["prob_b"]
    return prob_cache[key]


# ── Funciones privadas ────────────────────────────────────────────────────────

def _simulate_one(
    bracket: list[tuple[str, str]],
    active_rounds: list[str],
    ratings: dict[str, float],
    df_processed: pd.DataFrame,
    rng: np.random.Generator,
    prob_cache: dict[tuple[str, str], float],
    reach_count: dict[str, dict[str, int]],
) -> None:
    """
    Simula una iteración completa desde la ronda actual hasta el campeón.
    Acumula en reach_count los equipos que superan cada ronda.
    """
    current_pairs = list(bracket)

    for round_name in active_rounds[:-1]:   # excluye "Winner" del bucle
        winners = []
        for team_a, team_b in current_pairs:
            prob_a = get_cached_prob(team_a, team_b, ratings, df_processed, prob_cache)
            winner = team_a if rng.random() < prob_a else team_b
            reach_count[winner][round_name] += 1
            winners.append(winner)

        if round_name == "Final":
            reach_count[winners[0]]["Winner"] += 1
            break

        # Emparejar ganadores: (w0 vs w1), (w2 vs w3), …
        current_pairs = [
            (winners[i], winners[i + 1])
            for i in range(0, len(winners), 2)
        ]


def _build_dataframe(
    bracket: list[tuple[str, str]],
    ratings: dict[str, float],
    reach_count: dict[str, dict[str, int]],
    n_simulations: int,
    active_rounds: list[str],
) -> pd.DataFrame:
    """Construye el DataFrame con porcentajes por ronda simulada."""
    from src.elo_model import get_elo

    all_teams    = [team for pair in bracket for team in pair]
    output_rounds = active_rounds  # incluye "Winner"
    rows = []

    for team in all_teams:
        row = {
            "team": team,
            "elo":  round(get_elo(ratings, team), 1),
        }
        for round_name in output_rounds:
            pct = reach_count[team].get(round_name, 0) / n_simulations * 100
            row[round_name] = round(pct, 2)
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("Final", ascending=False)
        .reset_index(drop=True)
    )


def _validate_bracket(bracket: list[tuple[str, str]], starting_round: str) -> None:
    """Valida que el bracket tenga el tamaño correcto para la ronda indicada."""
    expected_sizes = {
        "Round of 32":    16,
        "Round of 16":    8,
        "Quarter-finals": 4,
        "Semi-finals":    2,
        "Final":          1,
    }
    if starting_round not in expected_sizes:
        raise ValueError(f"starting_round inválido: '{starting_round}'. "
                         f"Valores válidos: {list(expected_sizes.keys())}")
    expected = expected_sizes[starting_round]
    if len(bracket) != expected:
        raise ValueError(
            f"Para '{starting_round}' se esperan {expected} cruces, "
            f"pero se recibieron {len(bracket)}."
        )