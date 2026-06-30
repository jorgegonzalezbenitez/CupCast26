"""
api/main.py
===========
Responsabilidad única: exponer el modelo CupCast26 como API REST
con actualización automática cada 6 horas.

Endpoints
---------
GET  /                          Sirve la web (index.html)
GET  /api/status                Estado del servidor y última actualización
GET  /api/predictions/{round}   Predicciones de una ronda concreta
GET  /api/montecarlo            Resultados Montecarlo por ronda
GET  /api/elo                   Ranking Elo de los equipos activos
GET  /api/bracket               Estado del torneo (ronda actual + bracket)
GET  /api/accuracy              Precisión del modelo vs resultados reales
GET  /api/h2h/{team_a}/{team_b} Historial directo entre dos selecciones
POST /api/update                Fuerza actualización manual (requiere API key)

Arquitectura de caché
---------------------
Todo el estado del modelo vive en ``AppState``, un objeto singleton
que se recalcula en background cada 6 horas o bajo demanda.
Las requests sirven siempre desde caché → latencia < 5ms.

Scheduler
---------
APScheduler lanza ``refresh_model()`` automáticamente.
Al arrancar el servidor se ejecuta una primera carga síncrona
para que la API esté lista desde el primer request.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ── Asegurar que src/ y raíz son importables ─────────────────────────────────
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader         import load_raw_data
from src.feature_engineering import engineer_features, save_processed, get_h2h_stats
from src.elo_model           import compute_elo_ratings, elo_summary
from src.match_predictor     import predict_bracket
from src.montecarlo          import run_montecarlo, ROUND_LABELS
from api.accuracy            import extract_wc2026_results, compute_accuracy

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s · %(message)s")
logger = logging.getLogger(__name__)

# ── Configuración ─────────────────────────────────────────────────────────────

UPDATE_API_KEY   = os.getenv("CUPCAST_API_KEY", "cupcast2026")
N_SIMULATIONS    = int(os.getenv("N_SIMULATIONS", "50000"))
REFRESH_HOURS    = int(os.getenv("REFRESH_HOURS", "6"))

# ── Bracket y ronda actual ─────────────────────────────────────────────────────
# Actualizar RONDA_ACTUAL y BRACKET cuando avance el torneo

RONDA_ACTUAL = "Round of 32"

BRACKET: list[tuple[str, str]] = [
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


# ── Estado de la aplicación (caché en memoria) ────────────────────────────────

class AppState:
    """
    Singleton que almacena en memoria todos los resultados del modelo.
    Se recalcula en background — las requests nunca esperan al modelo.
    """
    def __init__(self):
        self.last_update:    Optional[datetime] = None
        self.is_ready:       bool               = False
        self.predictions:    dict               = {}
        self.montecarlo:     dict               = {}
        self.elo_ranking:    dict               = {}
        self.bracket_state:  dict               = {}
        self.accuracy:       dict               = {}
        self.pred_cache:     dict               = {}   # (team_a, team_b) → pred dict
        self.df_processed                       = None # para consultas H2H bajo demanda


state = AppState()


# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "CupCast26 API",
    description = "Predictor del Mundial 2026 mediante Elo + Ranking FIFA + Montecarlo",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Servir ficheros estáticos de la web
WEB_DIR = ROOT / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# ── Lógica de actualización del modelo ───────────────────────────────────────

async def refresh_model() -> None:
    """
    Descarga el CSV actualizado, recalcula el modelo completo y
    actualiza el estado en memoria.
    Llamado al arrancar y cada REFRESH_HOURS horas por el scheduler.
    """
    logger.info("═══ Iniciando actualización del modelo ═══")
    try:
        # 1. Datos frescos (force_download=True para obtener partidos nuevos)
        raw = load_raw_data(force_download=True)
        logger.info("    %d partidos cargados.", len(raw))

        # 2. Feature engineering
        processed = engineer_features(raw)
        save_processed(processed)

        # 3. Elo
        ratings   = compute_elo_ratings(processed)
        all_teams = list({t for pair in BRACKET for t in pair})
        df_elo    = elo_summary(ratings, all_teams)

        # 4. Predicciones por partido
        df_pred = predict_bracket(BRACKET, ratings, processed)

        # Caché de predicciones indexado por par de equipos
        pred_cache = {}
        for _, row in df_pred.iterrows():
            key_ab = (row["team_a"], row["team_b"])
            key_ba = (row["team_b"], row["team_a"])
            pred_dict = row.to_dict()
            pred_cache[key_ab] = pred_dict
            pred_cache[key_ba] = {
                **pred_dict,
                "team_a":   row["team_b"],
                "team_b":   row["team_a"],
                "prob_a":   row["prob_b"],
                "prob_b":   row["prob_a"],
                "favorite": row["team_b"] if row["prob_b"] > row["prob_a"] else row["team_a"],
            }

        # 5. Montecarlo
        df_mc = run_montecarlo(
            bracket        = BRACKET,
            ratings        = ratings,
            df_processed   = processed,
            n_simulations  = N_SIMULATIONS,
            seed           = 42,
            starting_round = RONDA_ACTUAL,
        )

        # 6. Accuracy (partidos ya jugados del Mundial 2026)
        wc_results = extract_wc2026_results(raw)
        report     = compute_accuracy(wc_results, pred_cache)

        # ── Actualizar estado ─────────────────────────────────────────────────
        round_cols = [c for c in df_mc.columns if c not in ("team", "elo")]

        state.predictions = {
            "round":   ROUND_LABELS.get(RONDA_ACTUAL, RONDA_ACTUAL),
            "matches": df_pred[[
                "team_a", "team_b", "prob_a", "prob_b",
                "favorite", "confidence",
                "elo_a", "elo_b", "p_elo_a", "p_fifa_a",
                "p_h2h_a", "h2h_matches", "h2h_used",
            ]].to_dict(orient="records"),
        }

        state.montecarlo = {
            "round_labels": [ROUND_LABELS.get(c, c) for c in round_cols],
            "teams": df_mc.to_dict(orient="records"),
        }

        state.elo_ranking = {
            "teams": df_elo[["team", "elo", "elo_rank"]].to_dict(orient="records"),
        }

        state.bracket_state = {
            "current_round":  ROUND_LABELS.get(RONDA_ACTUAL, RONDA_ACTUAL),
            "internal_round": RONDA_ACTUAL,
            "matches": [
                {"team_a": a, "team_b": b}
                for a, b in BRACKET
            ],
        }

        state.accuracy = {
            "overall_accuracy": report.overall_accuracy,
            "total_matches":    report.total_matches,
            "correct":          report.correct,
            "by_round":         report.by_round,
            "match_results": [
                {
                    "team_a":           m.team_a,
                    "team_b":           m.team_b,
                    "round":            m.round_name,
                    "prob_a":           m.prob_a,
                    "prob_b":           m.prob_b,
                    "predicted_winner": m.predicted_winner,
                    "real_winner":      m.real_winner,
                    "correct":          m.correct,
                    "confidence":       m.confidence,
                    "date":             m.date,
                }
                for m in report.match_results
            ],
        }

        state.pred_cache   = pred_cache
        state.df_processed = processed
        state.last_update  = datetime.utcnow()
        state.is_ready     = True

        logger.info("═══ Modelo actualizado correctamente · %s ═══", state.last_update)

    except Exception as exc:
        logger.exception("Error al actualizar el modelo: %s", exc)


# ── Eventos de ciclo de vida ──────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    """Al arrancar: carga inicial + scheduler cada REFRESH_HOURS horas."""
    await refresh_model()

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        refresh_model,
        trigger  = "interval",
        hours    = REFRESH_HOURS,
        id       = "refresh_model",
        replace_existing = True,
    )
    scheduler.start()
    logger.info("Scheduler activo: actualización cada %d horas.", REFRESH_HOURS)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    """Sirve la página principal de la web."""
    index = WEB_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "CupCast26 API · web/ no encontrada"}


@app.get("/api/status")
async def get_status():
    """Estado del servidor y metadatos de la última actualización."""
    return {
        "status":       "ready" if state.is_ready else "loading",
        "last_update":  state.last_update.isoformat() if state.last_update else None,
        "current_round": ROUND_LABELS.get(RONDA_ACTUAL, RONDA_ACTUAL),
        "n_simulations": N_SIMULATIONS,
        "refresh_hours": REFRESH_HOURS,
        "version":       "1.0.0",
    }


@app.get("/api/predictions")
async def get_predictions():
    """
    Predicciones de los partidos de la ronda actual.
    Incluye prob_a, prob_b, Elo, señales FIFA y H2H por partido.
    """
    _check_ready()
    return state.predictions


@app.get("/api/montecarlo")
async def get_montecarlo():
    """
    Resultados de la simulación Montecarlo (50 000 iteraciones).
    % de veces que cada selección alcanzó cada ronda.
    """
    _check_ready()
    return state.montecarlo


@app.get("/api/elo")
async def get_elo():
    """Ranking Elo de las selecciones clasificadas a la ronda actual."""
    _check_ready()
    return state.elo_ranking


@app.get("/api/bracket")
async def get_bracket():
    """Estado actual del bracket: ronda en curso y cruces."""
    _check_ready()
    return state.bracket_state


@app.get("/api/accuracy")
async def get_accuracy():
    """
    Precisión del modelo frente a resultados reales ya jugados.
    Accuracy global y desglosado por ronda.
    """
    _check_ready()
    return state.accuracy


@app.get("/api/h2h/{team_a}/{team_b}")
async def get_head_to_head(team_a: str, team_b: str):
    """
    Historial directo entre dos selecciones: partidos totales, % de
    victorias ponderado por recencia y los últimos 5 enfrentamientos
    con fecha, resultado y competición.
    """
    _check_ready()
    if state.df_processed is None:
        raise HTTPException(status_code=503, detail="Datos aún no disponibles.")
    return get_h2h_stats(state.df_processed, team_a, team_b)


@app.post("/api/update")
async def force_update(x_api_key: str = Header(default=None)):
    """
    Fuerza una actualización inmediata del modelo.
    Requiere la cabecera X-Api-Key con el valor correcto.

    Uso:
        curl -X POST https://tu-servidor/api/update -H "X-Api-Key: cupcast2026"
    """
    if x_api_key != UPDATE_API_KEY:
        raise HTTPException(status_code=401, detail="API key inválida.")
    await refresh_model()
    return {
        "status":      "updated",
        "last_update": state.last_update.isoformat(),
    }


# ── Helper ────────────────────────────────────────────────────────────────────

def _check_ready() -> None:
    """Lanza 503 si el modelo todavía no ha terminado la carga inicial."""
    if not state.is_ready:
        raise HTTPException(
            status_code = 503,
            detail      = "El modelo está cargando. Intenta en unos segundos.",
        )


# ── Ejecución directa ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)