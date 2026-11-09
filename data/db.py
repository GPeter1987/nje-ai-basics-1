"""
SQLite-elérés (foods, analysis_cache, history táblák).  Felelős: C
"""

import sqlite3

import config


def get_connection() -> sqlite3.Connection:
    """Kapcsolat a data/nutrilens.db adatbázishoz (sorok dict-szerűen érhetők el)."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
