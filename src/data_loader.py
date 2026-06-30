"""
data_loader.py
==============
Responsabilidad única: obtener el CSV crudo y devolverlo como DataFrame.

Fuente de datos
---------------
Repositorio público de martj42:
https://github.com/martj42/international_results

Columnas del CSV
----------------
date         : fecha del partido (YYYY-MM-DD)
home_team    : equipo local
away_team    : equipo visitante
home_score   : goles del equipo local
away_score   : goles del equipo visitante
tournament   : nombre de la competición
city         : ciudad donde se disputó
country      : país anfitrión
neutral      : True si el partido se jugó en campo neutral
"""

import logging
from pathlib import Path

import pandas as pd
import requests

# ── Configuración ─────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

DATA_URL = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/results.csv"
)

RAW_DIR = Path(__file__).parents[1] / "data" / "raw"
RAW_PATH = RAW_DIR / "results.csv"

REQUIRED_COLUMNS = {
    "date", "home_team", "away_team",
    "home_score", "away_score", "tournament", "neutral",
}


# ── Funciones públicas ────────────────────────────────────────────────────────

def download_raw_data(force: bool = False) -> Path:
    """
    Descarga el CSV desde el repositorio de martj42.

    Parameters
    ----------
    force : bool
        Si True, descarga aunque el fichero ya exista localmente.

    Returns
    -------
    Path
        Ruta al fichero descargado.

    Raises
    ------
    requests.HTTPError
        Si la descarga falla por un error HTTP.
    """
    if RAW_PATH.exists() and not force:
        logger.info("CSV ya disponible en %s — omitiendo descarga.", RAW_PATH)
        return RAW_PATH

    logger.info("Descargando datos desde %s …", DATA_URL)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()

    RAW_PATH.write_bytes(response.content)
    logger.info("CSV guardado en %s (%d bytes).", RAW_PATH, len(response.content))
    return RAW_PATH


def load_raw_data(force_download: bool = False) -> pd.DataFrame:
    """
    Carga el histórico de partidos internacionales como DataFrame.

    Descarga automáticamente el CSV si no existe en disco.

    Parameters
    ----------
    force_download : bool
        Fuerza la re-descarga aunque el fichero ya exista.

    Returns
    -------
    pd.DataFrame
        DataFrame ordenado cronológicamente con columnas tipadas.

    Raises
    ------
    ValueError
        Si el CSV no contiene las columnas mínimas esperadas.
    """
    path = download_raw_data(force=force_download)
    df = pd.read_csv(path, parse_dates=["date"])

    _validate_columns(df)
    df = _cast_types(df)
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(
        "Dataset cargado: %d partidos · %s → %s.",
        len(df),
        df["date"].min().date(),
        df["date"].max().date(),
    )
    return df


# ── Funciones privadas ────────────────────────────────────────────────────────

def _validate_columns(df: pd.DataFrame) -> None:
    """Lanza ValueError si faltan columnas obligatorias."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en el CSV: {missing}")


def _cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Garantiza los tipos correctos en cada columna."""
    df = df.copy()
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"] = df["neutral"].astype(bool)
    df["home_team"] = df["home_team"].str.strip()
    df["away_team"] = df["away_team"].str.strip()
    df["tournament"] = df["tournament"].str.strip()

    # Eliminar filas sin resultado (partidos futuros en el CSV)
    df = df.dropna(subset=["home_score", "away_score"])
    return df


# ── Ejecución directa (smoke test) ───────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s · %(message)s")
    data = load_raw_data()
    print(f"\nShape     : {data.shape}")
    print(f"Columnas  : {list(data.columns)}")
    print(f"Primeros  :\n{data.head(3).to_string()}")
    print(f"Últimos   :\n{data.tail(3).to_string()}")