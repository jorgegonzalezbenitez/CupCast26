"""
feature_engineering.py
======================
Responsabilidad única: enriquecer el DataFrame crudo con todas las
variables explicativas necesarias para el modelo de predicción.

Capas de features
-----------------
1. Peso por torneo       → tipo de competición (Mundial > clasif. > amistoso)
2. Peso temporal         → decaimiento exponencial hacia el pasado
3. Peso combinado        → producto de (1) y (2), alimenta el K dinámico del Elo
4. Resultado codificado  → perspectiva del local (1.0 / 0.5 / 0.0)
5. Ranking FIFA jun-2026 → puntos oficiales de las 32 selecciones de 16avos
                           como ancla de calidad absoluta actual
6. Calidad del rival     → el FIFA del rival pesa el resultado:
                           ganar a una potencia vale más que ganar a un débil
7. Head-to-head (H2H)    → historial directo ponderado por recencia entre
                           dos selecciones concretas (se calcula bajo demanda)

Diseño
------
- Las funciones públicas son: engineer_features, get_h2h_stats,
  get_fifa_points, save_processed, load_processed
- Todo lo privado (_) es interno al módulo
- Los nombres de selecciones siguen la nomenclatura del CSV de martj42
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── Logger ────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ── Rutas ─────────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path(__file__).parents[1] / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "matches_weighted.csv"

# ── Constantes temporales ─────────────────────────────────────────────────────

# Fecha de referencia: inicio del Mundial 2026
REFERENCE_DATE = pd.Timestamp("2026-06-11")

# Semivida del decaimiento exponencial
#   hace 0 años → peso 1.00
#   hace 4 años → peso 0.50
#   hace 8 años → peso 0.25
DECAY_HALF_LIFE_YEARS: float = 4.0

# ── Pesos por torneo ──────────────────────────────────────────────────────────
# Orden de prioridad: el primer match encontrado gana.
# Se busca por subcadena (case-insensitive).
TOURNAMENT_WEIGHTS: list[tuple[str, float]] = [
    ("FIFA World Cup qualification", 1.20),
    ("FIFA World Cup",               1.50),   # máxima importancia
    ("UEFA Euro qualification",      1.10),
    ("UEFA Euro",                    1.30),
    ("Copa América",                 1.30),
    ("AFC Asian Cup qualification",  1.10),
    ("AFC Asian Cup",                1.25),
    ("Africa Cup of Nations",        1.20),
    ("Confederations Cup",           1.15),
    ("UEFA Nations League",          1.15),
    ("CONCACAF Gold Cup",            1.15),
    ("Copa Africa",                  1.10),
    ("Friendly",                     0.80),   # menor importancia
]

DEFAULT_TOURNAMENT_WEIGHT: float = 1.00  # torneos oficiales no mapeados

# ── Ranking FIFA oficial — 11 junio 2026 ──────────────────────────────────────
# Fuente: FIFA/Coca-Cola Men's World Ranking (última actualización pre-Mundial)
# Incluye las 32 selecciones clasificadas a los 16avos de final.
# Puesto FIFA entre paréntesis para trazabilidad.
# Nombres según nomenclatura del CSV martj42.
FIFA_RANKING_2026: dict[str, dict] = {
    # ── Top potencias ──────────────────────────────────────────────────────
    "Argentina":               {"rank":  1, "points": 1877.27},
    "Spain":                   {"rank":  2, "points": 1874.71},
    "France":                  {"rank":  3, "points": 1870.70},
    "England":                 {"rank":  4, "points": 1828.02},
    "Portugal":                {"rank":  5, "points": 1767.85},
    "Brazil":                  {"rank":  6, "points": 1765.86},
    "Morocco":                 {"rank": 7, "points": 1755.10},
    "Netherlands":             {"rank":  8, "points": 1753.57},
    "Belgium":                 {"rank":  9, "points": 1742.24},
    "Germany":                 {"rank":  10, "points": 1735.77},
    # ── Selecciones de nivel medio-alto ────────────────────────────────────
    "Colombia":                {"rank": 13, "points": 1698.35},
    "Mexico":                  {"rank": 14, "points": 1687.48},
    "Croatia":                 {"rank": 11, "points": 1714.87},
    "Japan":                   {"rank": 18, "points": 1661.58},
    "Senegal":                 {"rank": 15, "points": 1684.07},
    "Switzerland":             {"rank": 19, "points": 1650.06},
    "Ecuador":                 {"rank": 23, "points": 1598.52},
    "Austria":                 {"rank": 24, "points": 1597.40},
    "Australia":               {"rank": 27, "points": 1579.34},
    "Egypt":                   {"rank": 29, "points": 1562.37},
    "Norway":                  {"rank": 31, "points": 1557.44},
    "Canada":                  {"rank": 30, "points": 1559.48},
    "Algeria":                 {"rank": 28, "points": 1571.03},
    "Ivory Coast":             {"rank": 33, "points": 1540.87},
    "United States":           {"rank": 17, "points": 1671.23},
    "Sweden":                  {"rank": 38, "points": 1509.79},
    "Paraguay":                {"rank": 41, "points": 1505.35},
    # ── Selecciones de nivel medio-bajo presentes en 16avos ───────────────
    "DR Congo":                {"rank": 46, "points": 1474.43},
    "South Africa":            {"rank": 60, "points": 1428.38},
    "Bosnia and Herzegovina":  {"rank": 64, "points": 1387.22},
    "Cape Verde":              {"rank": 67, "points": 1371.11},
    "Ghana":                   {"rank": 73, "points": 1346.88},
}

# Puntos FIFA para selecciones no listadas (media aproximada de la tabla)
FIFA_DEFAULT_POINTS: float = 1400.0

# Rango para normalización al intervalo [0, 1]
FIFA_POINTS_MIN: float = 1100.0
FIFA_POINTS_MAX: float = 1950.0


# ── Funciones públicas ────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el pipeline completo de feature engineering al DataFrame crudo.

    Columnas añadidas
    -----------------
    tournament_weight   float  Multiplicador según tipo de competición
    time_weight         float  Peso por antigüedad (decay exponencial)
    combined_weight     float  tournament_weight × time_weight
    home_result         float  1.0 gana local | 0.5 empate | 0.0 gana visitante
    fifa_points_home    float  Puntos FIFA del equipo local (jun-2026)
    fifa_points_away    float  Puntos FIFA del equipo visitante (jun-2026)
    fifa_diff           float  Diferencia de puntos FIFA (home − away)
    fifa_norm_home      float  Puntos FIFA local normalizados en [0, 1]
    fifa_norm_away      float  Puntos FIFA visitante normalizados en [0, 1]
    rival_quality_home  float  Calidad del RIVAL para el local (fifa_norm_away)
                               Ganar a un rival de alta calidad aporta más señal
    rival_quality_away  float  Calidad del RIVAL para el visitante (fifa_norm_home)

    Parameters
    ----------
    df : pd.DataFrame
        Salida de ``data_loader.load_raw_data()``, ordenado por fecha.

    Returns
    -------
    pd.DataFrame enriquecido con las columnas descritas.
    """
    df = df.copy()

    # Capa 1 · peso por torneo
    df["tournament_weight"] = df["tournament"].map(_get_tournament_weight)

    # Capa 2 · peso temporal
    df["time_weight"] = df["date"].map(_get_time_weight)

    # Capa 3 · peso combinado → input del K dinámico del Elo
    df["combined_weight"] = df["tournament_weight"] * df["time_weight"]

    # Capa 4 · resultado codificado desde perspectiva local
    df["home_result"] = df.apply(_encode_result, axis=1)

    # Capa 5 · ranking FIFA como ancla de calidad absoluta
    df["fifa_points_home"] = df["home_team"].map(
        lambda t: FIFA_RANKING_2026.get(t, {}).get("points", FIFA_DEFAULT_POINTS)
    )
    df["fifa_points_away"] = df["away_team"].map(
        lambda t: FIFA_RANKING_2026.get(t, {}).get("points", FIFA_DEFAULT_POINTS)
    )
    df["fifa_diff"]      = df["fifa_points_home"] - df["fifa_points_away"]
    df["fifa_norm_home"] = _normalize_fifa(df["fifa_points_home"])
    df["fifa_norm_away"] = _normalize_fifa(df["fifa_points_away"])

    # Capa 6 · calidad del rival
    # La señal de un resultado es más fuerte cuanto más potente es el rival.
    # rival_quality_home = fifa_norm del VISITANTE visto desde el local
    # rival_quality_away = fifa_norm del LOCAL visto desde el visitante
    df["rival_quality_home"] = df["fifa_norm_away"]
    df["rival_quality_away"] = df["fifa_norm_home"]

    logger.info(
        "Features generadas · %d partidos · "
        "peso combinado [min=%.4f | max=%.4f | media=%.4f]",
        len(df),
        df["combined_weight"].min(),
        df["combined_weight"].max(),
        df["combined_weight"].mean(),
    )
    return df


def get_h2h_stats(
    df: pd.DataFrame,
    team_a: str,
    team_b: str,
    min_matches: int = 3,
) -> dict:
    """
    Calcula el historial directo (head-to-head) entre dos selecciones,
    ponderado por recencia (combined_weight).

    El H2H recoge TODOS los enfrentamientos históricos entre ambas selecciones,
    independientemente de quién fue local. Los resultados se expresan siempre
    desde la perspectiva de team_a.

    El peso temporal hace que los enfrentamientos recientes dominen la señal:
    un clásico de hace 15 años contribuye, pero apenas mueve la aguja frente
    a un partido de Nations League del año pasado.

    Parameters
    ----------
    df : pd.DataFrame
        Salida de ``engineer_features()``.
    team_a : str
        Nombre de la primera selección (nomenclatura del CSV martj42).
    team_b : str
        Nombre de la segunda selección.
    min_matches : int
        Avisa por log si hay menos partidos que este umbral.

    Returns
    -------
    dict con claves:
        team_a          str    Nombre de la selección A
        team_b          str    Nombre de la selección B
        total_matches   int    Partidos disputados en total
        win_rate_a      float  % victorias ponderadas de A (0–1)
        win_rate_b      float  % victorias ponderadas de B (0–1)
        draw_rate       float  % empates ponderados (0–1)
        avg_goals_a     float  Goles por partido de A (ponderado)
        avg_goals_b     float  Goles por partido de B (ponderado)
        goal_diff_a     float  Diferencia de goles media a favor de A
        h2h_advantage   str    "A", "B" o "EVEN" según quién domina el H2H
        last_5          list   Últimos 5 enfrentamientos, cada uno:
                               {'date', 'score', 'result_for_a', 'tournament'}
    """
    h2h = _filter_h2h(df, team_a, team_b)

    if h2h.empty:
        logger.warning("Sin historial H2H entre %s y %s.", team_a, team_b)
        return _empty_h2h(team_a, team_b)

    if len(h2h) < min_matches:
        logger.warning(
            "Solo %d partido(s) H2H entre %s y %s — estadísticas poco robustas.",
            len(h2h), team_a, team_b,
        )

    goals_a  = h2h["goals_a"].values
    goals_b  = h2h["goals_b"].values
    weights  = h2h["combined_weight"].values
    total_w  = weights.sum()

    wins_a   = (goals_a > goals_b).astype(float)
    wins_b   = (goals_b > goals_a).astype(float)
    draws    = (goals_a == goals_b).astype(float)

    win_rate_a  = float(np.dot(wins_a,  weights) / total_w)
    win_rate_b  = float(np.dot(wins_b,  weights) / total_w)
    draw_rate   = float(np.dot(draws,   weights) / total_w)
    avg_goals_a = float(np.dot(goals_a, weights) / total_w)
    avg_goals_b = float(np.dot(goals_b, weights) / total_w)

    # Ventaja histórica: diferencia de win_rate superior a 5 pp
    if win_rate_a - win_rate_b > 0.05:
        advantage = "A"
    elif win_rate_b - win_rate_a > 0.05:
        advantage = "B"
    else:
        advantage = "EVEN"

    last_5 = (
        h2h.sort_values("date", ascending=False)
        .head(5)
        .apply(
            lambda row: {
                "date":         str(row["date"].date()),
                "score":        f"{int(row['goals_a'])}-{int(row['goals_b'])}",
                "result_for_a": (
                    "W" if row["goals_a"] > row["goals_b"] else
                    "L" if row["goals_a"] < row["goals_b"] else "D"
                ),
                "tournament":   row["tournament"],
            },
            axis=1,
        )
        .tolist()
    )

    return {
        "team_a":        team_a,
        "team_b":        team_b,
        "total_matches": len(h2h),
        "win_rate_a":    round(win_rate_a,  4),
        "win_rate_b":    round(win_rate_b,  4),
        "draw_rate":     round(draw_rate,   4),
        "avg_goals_a":   round(avg_goals_a, 2),
        "avg_goals_b":   round(avg_goals_b, 2),
        "goal_diff_a":   round(avg_goals_a - avg_goals_b, 2),
        "h2h_advantage": advantage,
        "last_5":        last_5,
    }


def get_fifa_points(team: str) -> float:
    """Devuelve los puntos FIFA de una selección o el valor por defecto."""
    return FIFA_RANKING_2026.get(team, {}).get("points", FIFA_DEFAULT_POINTS)


def get_fifa_rank(team: str) -> int:
    """Devuelve el puesto FIFA de una selección o 999 si no está en la tabla."""
    return FIFA_RANKING_2026.get(team, {}).get("rank", 999)


def save_processed(df: pd.DataFrame) -> Path:
    """
    Persiste el DataFrame procesado en data/processed/matches_weighted.csv.

    Returns
    -------
    Path al fichero guardado.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    logger.info("Dataset procesado guardado en %s.", PROCESSED_PATH)
    return PROCESSED_PATH


def load_processed() -> pd.DataFrame:
    """
    Carga el CSV procesado desde disco.

    Raises
    ------
    FileNotFoundError si el CSV procesado no existe todavía.
    """
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {PROCESSED_PATH}. "
            "Ejecuta primero engineer_features() y save_processed()."
        )
    return pd.read_csv(PROCESSED_PATH, parse_dates=["date"])


# ── Funciones privadas ────────────────────────────────────────────────────────

def _get_tournament_weight(tournament: str) -> float:
    """Devuelve el multiplicador del torneo recorriendo la lista en orden."""
    t_lower = tournament.lower()
    for keyword, weight in TOURNAMENT_WEIGHTS:
        if keyword.lower() in t_lower:
            return weight
    return DEFAULT_TOURNAMENT_WEIGHT


def _get_time_weight(date: pd.Timestamp) -> float:
    """
    Decay exponencial: w = exp(-ln(2) × años_de_antigüedad / semivida)
    Partidos futuros (fecha > REFERENCE_DATE) reciben peso máximo = 1.0.
    """
    years_ago = max((REFERENCE_DATE - date).days / 365.25, 0.0)
    return float(np.exp(-np.log(2) * years_ago / DECAY_HALF_LIFE_YEARS))


def _encode_result(row: pd.Series) -> float:
    """1.0 local gana | 0.5 empate | 0.0 visitante gana."""
    if row["home_score"] > row["away_score"]:
        return 1.0
    if row["home_score"] < row["away_score"]:
        return 0.0
    return 0.5


def _normalize_fifa(series: pd.Series) -> pd.Series:
    """Min-max normalización de puntos FIFA al rango [0, 1]."""
    return (series - FIFA_POINTS_MIN) / (FIFA_POINTS_MAX - FIFA_POINTS_MIN)


def _filter_h2h(
    df: pd.DataFrame, team_a: str, team_b: str
) -> pd.DataFrame:
    """
    Filtra todos los partidos entre team_a y team_b.
    Estandariza goles desde la perspectiva de team_a independientemente
    de quién jugó como local en cada partido.
    """
    mask_a_home = (df["home_team"] == team_a) & (df["away_team"] == team_b)
    mask_b_home = (df["home_team"] == team_b) & (df["away_team"] == team_a)

    cols = ["date", "home_score", "away_score", "tournament", "combined_weight"]

    part_a = df[mask_a_home][cols].rename(
        columns={"home_score": "goals_a", "away_score": "goals_b"}
    )
    part_b = df[mask_b_home][cols].rename(
        columns={"away_score": "goals_a", "home_score": "goals_b"}
    )

    return (
        pd.concat([part_a, part_b], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )


def _empty_h2h(team_a: str, team_b: str) -> dict:
    """Devuelve un dict H2H neutral cuando no hay historial entre los equipos."""
    return {
        "team_a": team_a, "team_b": team_b,
        "total_matches": 0,
        "win_rate_a":    0.5,  "win_rate_b": 0.5, "draw_rate": 0.0,
        "avg_goals_a":   0.0,  "avg_goals_b": 0.0, "goal_diff_a": 0.0,
        "h2h_advantage": "EVEN",
        "last_5": [],
    }


# ── Ejecución directa (smoke test) ───────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1]))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s · %(message)s")

    from src.data_loader import load_raw_data

    raw       = load_raw_data()
    processed = engineer_features(raw)
    save_processed(processed)

    # ── Features de los últimos partidos
    print("\n── Últimas filas con features ──")
    cols = [
        "date", "home_team", "away_team",
        "tournament_weight", "time_weight", "combined_weight",
        "home_result", "fifa_diff", "rival_quality_home",
    ]
    print(processed[cols].tail(8).to_string(index=False))

    # ── H2H de los 16 cruces reales del Mundial 2026
    BRACKET_R16 = [
        ("South Africa",          "Canada"),
        ("Brazil",                "Japan"),
        ("Germany",               "Paraguay"),
        ("Netherlands",           "Morocco"),
        ("Ivory Coast",           "Norway"),
        ("France",                "Sweden"),
        ("Mexico",                "Ecuador"),
        ("England",               "DR Congo"),
        ("Belgium",               "Senegal"),
        ("United States",         "Bosnia and Herzegovina"),
        ("Spain",                 "Austria"),
        ("Portugal",              "Croatia"),
        ("Switzerland",           "Algeria"),
        ("Australia",             "Egypt"),
        ("Argentina",             "Cape Verde"),
        ("Colombia",              "Ghana"),
    ]

    print("\n── Head-to-head de los 16avos de final ──")
    print(f"{'Cruce':<40} {'Partidos':>8} {'W_A':>7} {'W_B':>7} {'GDif_A':>7} {'Ventaja':>9}")
    print("─" * 80)
    for a, b in BRACKET_R16:
        h = get_h2h_stats(processed, a, b)
        cruce = f"{a} vs {b}"
        print(
            f"{cruce:<40} {h['total_matches']:>8} "
            f"{h['win_rate_a']:>7.3f} {h['win_rate_b']:>7.3f} "
            f"{h['goal_diff_a']:>7.2f} {h['h2h_advantage']:>9}"
        )

    # ── Ranking FIFA de las 32 selecciones
    print("\n── Ranking FIFA 11-jun-2026 (32 selecciones de 16avos) ──")
    sorted_fifa = sorted(FIFA_RANKING_2026.items(), key=lambda x: x[1]["rank"])
    for team, data in sorted_fifa:
        print(f"  {data['rank']:>3}. {team:<30} {data['points']:.2f} pts")