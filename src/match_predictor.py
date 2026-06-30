"""
match_predictor.py
==================
Responsabilidad única: combinar las tres señales del modelo en una
probabilidad final de victoria para cada enfrentamiento.

Las tres señales
----------------
1. Elo histórico     → trayectoria real partido a partido desde 1872
                       con K dinámico (torneo × recencia × calidad rival)
2. Ranking FIFA      → calidad absoluta actual (jun-2026), ancla externa
                       que corrige sesgos del Elo en selecciones con poco
                       historial reciente
3. Head-to-head      → ventaja psicológica e histórica entre estos dos
                       equipos concretos, ponderada por recencia

Combinación
-----------
La probabilidad final es una media ponderada de las tres señales:

    P_final(A) = w_elo   × P_elo(A)
               + w_fifa  × P_fifa(A)
               + w_h2h   × P_h2h(A)

    con w_elo + w_fifa + w_h2h = 1.0

Pesos por defecto
-----------------
    ELO_WEIGHT  = 0.50  → señal más robusta: 49 000+ partidos históricos
    FIFA_WEIGHT = 0.30  → ancla de calidad actual muy relevante en Mundiales
    H2H_WEIGHT  = 0.20  → historial directo, pero solo si hay ≥ MIN_H2H_MATCHES
                          Si no hay H2H suficiente, su peso se redistribuye
                          proporcionalmente entre Elo y FIFA.

Ajuste por H2H insuficiente
----------------------------
Cuando dos selecciones tienen menos de MIN_H2H_MATCHES enfrentamientos
históricos, el H2H no es estadísticamente fiable. En ese caso el peso
de H2H se reparte entre Elo y FIFA manteniendo su proporción relativa:

    w_elo_adj  = ELO_WEIGHT  / (ELO_WEIGHT + FIFA_WEIGHT) × (1 − 0)
    w_fifa_adj = FIFA_WEIGHT / (ELO_WEIGHT + FIFA_WEIGHT) × (1 − 0)

Salida
------
``predict_match`` devuelve un dict con todas las señales intermedias
y la probabilidad final, lo que permite auditar cada predicción.
"""

import logging
from pathlib import Path

import pandas as pd

from src.elo_model           import get_elo, elo_win_probability
from src.feature_engineering import get_h2h_stats, get_fifa_points, FIFA_POINTS_MIN, FIFA_POINTS_MAX

# ── Logger ────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ── Pesos de combinación ──────────────────────────────────────────────────────

ELO_WEIGHT:  float = 0.50
FIFA_WEIGHT: float = 0.30
H2H_WEIGHT:  float = 0.20

# Mínimo de partidos H2H para considerar esa señal estadísticamente válida
MIN_H2H_MATCHES: int = 3


# ── Función pública principal ─────────────────────────────────────────────────

def predict_match(
    team_a: str,
    team_b: str,
    ratings: dict[str, float],
    df_processed: pd.DataFrame,
) -> dict:
    """
    Predice la probabilidad de victoria de cada equipo en campo neutral.

    Combina Elo histórico, ranking FIFA y head-to-head en una única
    probabilidad final ponderada.

    Parameters
    ----------
    team_a : str
        Nombre del primer equipo (nomenclatura CSV martj42).
    team_b : str
        Nombre del segundo equipo.
    ratings : dict[str, float]
        Salida de ``elo_model.compute_elo_ratings()``.
    df_processed : pd.DataFrame
        Salida de ``feature_engineering.engineer_features()``.
        Necesario para calcular el H2H.

    Returns
    -------
    dict con claves:
        team_a          str    Nombre equipo A
        team_b          str    Nombre equipo B

        elo_a           float  Rating Elo de A
        elo_b           float  Rating Elo de B
        p_elo_a         float  P(A gana) según Elo

        fifa_pts_a      float  Puntos FIFA de A
        fifa_pts_b      float  Puntos FIFA de B
        p_fifa_a        float  P(A gana) según ranking FIFA

        h2h_matches     int    Partidos H2H históricos
        h2h_win_rate_a  float  Win rate ponderado de A en el H2H
        p_h2h_a         float  P(A gana) según H2H (o 0.5 si insuficiente)
        h2h_used        bool   Si el H2H fue incluido en el cálculo final

        w_elo           float  Peso efectivo del Elo en la combinación
        w_fifa          float  Peso efectivo del FIFA en la combinación
        w_h2h           float  Peso efectivo del H2H en la combinación

        prob_a          float  Probabilidad final de victoria de A
        prob_b          float  Probabilidad final de victoria de B (= 1 − prob_a)
        favorite        str    Nombre del favorito
        confidence      str    "Alta" / "Media" / "Baja" según margen
    """
    # ── Señal 1: Elo ──────────────────────────────────────────────────────────
    elo_a = get_elo(ratings, team_a)
    elo_b = get_elo(ratings, team_b)
    p_elo_a = elo_win_probability(elo_a, elo_b)

    # ── Señal 2: Ranking FIFA ─────────────────────────────────────────────────
    fifa_a  = get_fifa_points(team_a)
    fifa_b  = get_fifa_points(team_b)
    p_fifa_a = _fifa_to_probability(fifa_a, fifa_b)

    # ── Señal 3: Head-to-head ─────────────────────────────────────────────────
    h2h      = get_h2h_stats(df_processed, team_a, team_b)
    h2h_used = h2h["total_matches"] >= MIN_H2H_MATCHES
    p_h2h_a  = h2h["win_rate_a"] if h2h_used else 0.5

    # ── Pesos efectivos ───────────────────────────────────────────────────────
    w_elo, w_fifa, w_h2h = _effective_weights(h2h_used)

    # ── Probabilidad final ────────────────────────────────────────────────────
    prob_a = w_elo * p_elo_a + w_fifa * p_fifa_a + w_h2h * p_h2h_a
    prob_b = 1.0 - prob_a

    favorite   = team_a if prob_a >= prob_b else team_b
    confidence = _confidence_label(abs(prob_a - prob_b))

    logger.debug(
        "%s vs %s → Elo %.3f | FIFA %.3f | H2H %.3f → FINAL %.3f",
        team_a, team_b, p_elo_a, p_fifa_a, p_h2h_a, prob_a,
    )

    return {
        # identidad
        "team_a":         team_a,
        "team_b":         team_b,
        # señal Elo
        "elo_a":          round(elo_a,    2),
        "elo_b":          round(elo_b,    2),
        "p_elo_a":        round(p_elo_a,  4),
        # señal FIFA
        "fifa_pts_a":     round(fifa_a,   2),
        "fifa_pts_b":     round(fifa_b,   2),
        "p_fifa_a":       round(p_fifa_a, 4),
        # señal H2H
        "h2h_matches":    h2h["total_matches"],
        "h2h_win_rate_a": round(h2h["win_rate_a"], 4),
        "h2h_goal_diff":  round(h2h["goal_diff_a"], 2),
        "p_h2h_a":        round(p_h2h_a,  4),
        "h2h_used":       h2h_used,
        # pesos efectivos
        "w_elo":          round(w_elo,  3),
        "w_fifa":         round(w_fifa, 3),
        "w_h2h":          round(w_h2h,  3),
        # resultado final
        "prob_a":         round(prob_a, 4),
        "prob_b":         round(prob_b, 4),
        "favorite":       favorite,
        "confidence":     confidence,
    }


def predict_bracket(
    bracket: list[tuple[str, str]],
    ratings: dict[str, float],
    df_processed: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predice todos los partidos de un bracket y devuelve un DataFrame resumen.

    Parameters
    ----------
    bracket : list[tuple[str, str]]
        Lista de cruces [(equipo_a, equipo_b), ...].
    ratings : dict[str, float]
        Salida de ``elo_model.compute_elo_ratings()``.
    df_processed : pd.DataFrame
        Salida de ``feature_engineering.engineer_features()``.

    Returns
    -------
    pd.DataFrame ordenado por prob_a descendente con columnas clave.
    """
    rows = []
    for team_a, team_b in bracket:
        pred = predict_match(team_a, team_b, ratings, df_processed)
        rows.append({
            "team_a":     pred["team_a"],
            "team_b":     pred["team_b"],
            "prob_a":     pred["prob_a"],
            "prob_b":     pred["prob_b"],
            "favorite":   pred["favorite"],
            "confidence": pred["confidence"],
            "elo_a":      pred["elo_a"],
            "elo_b":      pred["elo_b"],
            "p_elo_a":    pred["p_elo_a"],
            "p_fifa_a":   pred["p_fifa_a"],
            "p_h2h_a":    pred["p_h2h_a"],
            "h2h_matches":pred["h2h_matches"],
            "h2h_used":   pred["h2h_used"],
        })
    return (
        pd.DataFrame(rows)
        .sort_values("prob_a", ascending=False)
        .reset_index(drop=True)
    )


# ── Funciones privadas ────────────────────────────────────────────────────────

def _fifa_to_probability(fifa_a: float, fifa_b: float) -> float:
    """
    Convierte los puntos FIFA de dos equipos en P(A gana) usando la
    misma fórmula logística que el Elo pero escalada al rango FIFA.

    Normaliza ambos al rango [0, 1] y aplica softmax con temperatura T.
    Con T = 0.3 el modelo es más sensible a diferencias de ranking FIFA.
    """
    norm_a = (fifa_a - FIFA_POINTS_MIN) / (FIFA_POINTS_MAX - FIFA_POINTS_MIN)
    norm_b = (fifa_b - FIFA_POINTS_MIN) / (FIFA_POINTS_MAX - FIFA_POINTS_MIN)
    T = 0.3
    exp_a = 2.718281828 ** (norm_a / T)
    exp_b = 2.718281828 ** (norm_b / T)
    return exp_a / (exp_a + exp_b)


def _effective_weights(h2h_used: bool) -> tuple[float, float, float]:
    """
    Devuelve los pesos efectivos (w_elo, w_fifa, w_h2h).

    Si el H2H no es suficientemente robusto, su peso se redistribuye
    proporcionalmente entre Elo y FIFA.
    """
    if h2h_used:
        return ELO_WEIGHT, FIFA_WEIGHT, H2H_WEIGHT

    # Redistribución proporcional sin H2H
    total = ELO_WEIGHT + FIFA_WEIGHT
    w_elo  = ELO_WEIGHT  / total
    w_fifa = FIFA_WEIGHT / total
    return w_elo, w_fifa, 0.0


def _confidence_label(margin: float) -> str:
    """
    Etiqueta de confianza según el margen entre prob_a y prob_b.

        margin ≥ 0.25  →  Alta    (el favorito gana >62.5% de las veces)
        margin ≥ 0.10  →  Media   (el favorito gana >55%)
        margin <  0.10  →  Baja    (partido muy igualado)
    """
    if margin >= 0.25:
        return "Alta"
    if margin >= 0.10:
        return "Media"
    return "Baja"


# ── Ejecución directa (smoke test) ───────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1]))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s · %(message)s")

    from src.data_loader         import load_raw_data
    from src.feature_engineering import engineer_features
    from src.elo_model           import compute_elo_ratings

    raw       = load_raw_data()
    processed = engineer_features(raw)
    ratings   = compute_elo_ratings(processed)

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

    # ── Predicción detallada de un partido
    print("\n── Análisis detallado: Brasil vs Japón ──")
    det = predict_match("Brazil", "Japan", ratings, processed)
    for k, v in det.items():
        if k != "last_5":
            print(f"  {k:<20}: {v}")

    # ── Resumen de todos los cruces
    print("\n── Predicciones 16avos de final ──")
    df_pred = predict_bracket(BRACKET_R16, ratings, processed)
    cols = ["team_a", "team_b", "prob_a", "prob_b",
            "favorite", "confidence", "p_elo_a", "p_fifa_a", "p_h2h_a", "h2h_used"]
    print(df_pred[cols].to_string(index=False))