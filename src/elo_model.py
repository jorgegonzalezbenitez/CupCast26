"""
elo_model.py
============
Responsabilidad única: calcular el rating Elo de cada selección recorriendo
el histórico completo partido a partido (1872 → 2026) con K dinámico.

Fundamento matemático
---------------------
El sistema Elo estima la probabilidad de victoria del equipo A contra B:

    P(A gana) = 1 / (1 + 10^((Elo_B − Elo_A − ventaja_local) / 400))

Tras cada partido los ratings se actualizan simétricamente:

    ΔElo     = K × (resultado_real − P(A gana))
    Elo_A   += ΔElo
    Elo_B   -= ΔElo

K dinámico
----------
El factor K controla cuánto mueve cada partido el rating.
Aquí es dinámico: absorbe en un solo escalar toda la información
ya calculada en feature_engineering.py:

    K = K_BASE × combined_weight × rival_quality_modifier

    combined_weight       = tournament_weight × time_weight
                            (ya calculado en feature_engineering)
    rival_quality_modifier = 1 + RIVAL_QUALITY_SCALE × fifa_norm_rival
                            Ganar a Argentina (fifa_norm ≈ 0.95) mueve
                            más el rating que ganar a Cabo Verde (≈ 0.27)

Decisiones de diseño
--------------------
- K_BASE = 30    equilibrio clásico entre estabilidad y reactividad
- ELO_INICIAL = 1500 para todas las selecciones en su debut histórico
- VENTAJA_LOCAL = 80 pts  solo aplica cuando neutral=False
  (todos los partidos del Mundial 2026 son en campo neutral)
- RIVAL_QUALITY_SCALE = 0.5  el rival más potente posible multiplica
  K por 1.5; el más débil apenas lo toca (×1.0)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── Logger ────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ── Hiperparámetros del modelo ────────────────────────────────────────────────

ELO_INICIAL: float         = 1500.0
K_BASE: float              = 30.0
VENTAJA_LOCAL: float       = 80.0
RIVAL_QUALITY_SCALE: float = 0.5   # amplificación máxima del K por calidad rival

# Columnas mínimas requeridas en el DataFrame de entrada
_REQUIRED_COLS = {
    "home_team", "away_team", "home_result",
    "combined_weight", "neutral",
    "rival_quality_home", "rival_quality_away",
}


# ── Funciones públicas ────────────────────────────────────────────────────────

def compute_elo_ratings(df: pd.DataFrame) -> dict[str, float]:
    """
    Recorre el histórico cronológicamente y devuelve el rating Elo
    final de cada selección.

    El K dinámico integra tres señales en un solo escalar:
        1. Importancia del torneo   (tournament_weight)
        2. Recencia del partido     (time_weight)
        3. Calidad del rival        (rival_quality_home / rival_quality_away)

    Parameters
    ----------
    df : pd.DataFrame
        Salida de ``engineer_features()``, ordenado por fecha ascendente.
        Debe contener las columnas definidas en _REQUIRED_COLS.

    Returns
    -------
    dict[str, float]
        {nombre_selección: rating_elo_final}
    """
    _validate_columns(df)

    ratings: dict[str, float] = {}
    n = len(df)

    logger.info("Calculando Elo sobre %d partidos …", n)

    for i, row in enumerate(df.itertuples(index=False), start=1):

        home = row.home_team
        away = row.away_team

        elo_h = ratings.get(home, ELO_INICIAL)
        elo_a = ratings.get(away, ELO_INICIAL)

        neutral = bool(row.neutral)

        # ── K dinámico ───────────────────────────────────────────────────────
        # combined_weight ya integra torneo × tiempo
        # rival_quality_modifier amplifica K cuando el rival es más potente
        k_home = K_BASE * row.combined_weight * _rival_modifier(row.rival_quality_home)
        k_away = K_BASE * row.combined_weight * _rival_modifier(row.rival_quality_away)
        # K simétrico: promedio de ambas perspectivas para mantener suma cero
        k = (k_home + k_away) / 2.0

        # ── Actualización Elo ─────────────────────────────────────────────────
        prob_home = _expected_score(elo_h, elo_a, neutral)
        delta     = k * (row.home_result - prob_home)

        ratings[home] = elo_h + delta
        ratings[away] = elo_a - delta

        if i % 10_000 == 0:
            logger.debug("  Procesados %d / %d partidos …", i, n)

    logger.info(
        "Elo calculado para %d selecciones · rango [%.1f – %.1f]",
        len(ratings),
        min(ratings.values()),
        max(ratings.values()),
    )
    return ratings


def ratings_to_dataframe(ratings: dict[str, float]) -> pd.DataFrame:
    """
    Convierte el dict de ratings en un DataFrame ordenado por Elo descendente.

    Returns
    -------
    pd.DataFrame con columnas [rank, team, elo]
    """
    df = (
        pd.DataFrame(ratings.items(), columns=["team", "elo"])
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )
    df.index     += 1
    df.index.name = "rank"
    df["elo"]     = df["elo"].round(2)
    return df


def get_elo(ratings: dict[str, float], team: str) -> float:
    """
    Devuelve el Elo de una selección.
    Si no tiene historial, devuelve ELO_INICIAL (1500).

    Parameters
    ----------
    ratings : dict[str, float]
        Salida de ``compute_elo_ratings()``.
    team : str
        Nombre de la selección.
    """
    return ratings.get(team, ELO_INICIAL)


def elo_win_probability(elo_a: float, elo_b: float) -> float:
    """
    Calcula P(gana A) en campo neutral dados sus ratings Elo.

    Fórmula estándar con divisor 400:
        P = 1 / (1 + 10^((Elo_B − Elo_A) / 400))

    Parameters
    ----------
    elo_a : float  Rating Elo de la selección A
    elo_b : float  Rating Elo de la selección B

    Returns
    -------
    float en [0, 1]
    """
    return _expected_score(elo_a, elo_b, neutral=True)


def elo_summary(
    ratings: dict[str, float],
    teams: list[str],
) -> pd.DataFrame:
    """
    Devuelve un DataFrame con el Elo y ranking interno de una lista de equipos.
    Útil para comparar las selecciones clasificadas a una ronda concreta.

    Parameters
    ----------
    ratings : dict[str, float]
    teams   : list[str]  Lista de selecciones a comparar

    Returns
    -------
    pd.DataFrame con columnas [team, elo, elo_rank]
    """
    rows = [{"team": t, "elo": round(get_elo(ratings, t), 2)} for t in teams]
    df   = pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)
    df["elo_rank"] = df.index + 1
    return df


# ── Funciones privadas ────────────────────────────────────────────────────────

def _expected_score(elo_a: float, elo_b: float, neutral: bool) -> float:
    """
    Probabilidad esperada de victoria del equipo A (local) según Elo.
    Si neutral=False, se añade VENTAJA_LOCAL al Elo del local.
    """
    advantage = 0.0 if neutral else VENTAJA_LOCAL
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a - advantage) / 400.0))


def _rival_modifier(rival_quality_norm: float) -> float:
    """
    Modificador de K según la calidad normalizada del rival [0, 1].

    Fórmula lineal:
        modifier = 1 + RIVAL_QUALITY_SCALE × rival_quality_norm

    Ejemplos con RIVAL_QUALITY_SCALE = 0.5:
        rival_quality_norm = 0.00  (rival muy débil)  → modifier = 1.00
        rival_quality_norm = 0.50  (rival de nivel medio) → modifier = 1.25
        rival_quality_norm = 1.00  (rival élite)      → modifier = 1.50
    """
    return 1.0 + RIVAL_QUALITY_SCALE * float(rival_quality_norm)


def _validate_columns(df: pd.DataFrame) -> None:
    """Lanza ValueError si faltan columnas necesarias para el cálculo Elo."""
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas en el DataFrame para calcular el Elo: {missing}. "
            "Asegúrate de pasar la salida de engineer_features()."
        )


# ── Ejecución directa (smoke test) ───────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1]))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s · %(message)s")

    from src.data_loader        import load_raw_data
    from src.feature_engineering import engineer_features

    raw      = load_raw_data()
    processed = engineer_features(raw)
    ratings   = compute_elo_ratings(processed)
    df_elo    = ratings_to_dataframe(ratings)

    # ── Top 20 global
    print("\n── Top 20 selecciones por Elo ──")
    print(df_elo.head(20).to_string())

    # ── Elo de las 32 selecciones de 16avos
    R16_TEAMS = [
        "South Africa", "Canada", "Brazil", "Japan",
        "Germany", "Paraguay", "Netherlands", "Morocco",
        "Ivory Coast", "Norway", "France", "Sweden",
        "Mexico", "Ecuador", "England", "DR Congo",
        "Belgium", "Senegal", "United States", "Bosnia and Herzegovina",
        "Spain", "Austria", "Portugal", "Croatia",
        "Switzerland", "Algeria", "Australia", "Egypt",
        "Argentina", "Cape Verde", "Colombia", "Ghana",
    ]

    print("\n── Elo de las 32 selecciones clasificadas ──")
    df_r16 = elo_summary(ratings, R16_TEAMS)
    print(df_r16.to_string(index=False))

    # ── Probabilidades de los 16 cruces reales
    BRACKET_R16 = [
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
    ]

    print("\n── Probabilidades Elo — 16avos de final ──")
    print(f"  {'Partido':<42} {'Elo_A':>7} {'Elo_B':>7} {'P(A)':>7} {'P(B)':>7}")
    print("  " + "─" * 70)
    for a, b in BRACKET_R16:
        ea  = get_elo(ratings, a)
        eb  = get_elo(ratings, b)
        p_a = elo_win_probability(ea, eb)
        print(
            f"  {a+' vs '+b:<42} "
            f"{ea:>7.1f} {eb:>7.1f} "
            f"{p_a:>7.4f} {1-p_a:>7.4f}"
        )