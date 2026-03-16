"""
core/database.py
"""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class UserRow:
    id:         int
    name:       str
    created_at: str


@dataclass
class CaptureRow:
    id:         int
    user_id:    int
    image_path: str
    created_at: str


@dataclass
class AttackRow:
    id:                  int
    attacker_name:       str
    model_name:          str
    similarity_score:    float
    image_path:          str
    identity_name:       str
    matched_image_path:  str
    created_at:          str


class Database:
    def __init__(self, db_path: str = "db.sqlite") -> None:
        self._path = str(Path(db_path))
        self._init_schema()

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(self._path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _init_schema(self) -> None:
        # Create tables (new installs get all columns from the start)
        with self._connect() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    name       TEXT    NOT NULL UNIQUE,
                    created_at TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS captures (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    image_path TEXT    NOT NULL,
                    created_at TEXT    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS attacks (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    attacker_name       TEXT    NOT NULL,
                    model_name          TEXT    NOT NULL,
                    similarity_score    REAL    NOT NULL,
                    image_path          TEXT    NOT NULL,
                    identity_name       TEXT    NOT NULL DEFAULT '',
                    matched_image_path  TEXT    NOT NULL DEFAULT '',
                    created_at          TEXT    NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_attacks_model
                    ON attacks (model_name, similarity_score ASC);
            """)

        # Migrations: add new columns to existing databases that predate them.
        # Each ALTER TABLE runs in its own connection so a "duplicate column"
        # error from an already-migrated DB doesn't block the others.
        for col in ("identity_name", "matched_image_path"):
            try:
                with self._connect() as con:
                    con.execute(
                        f"ALTER TABLE attacks ADD COLUMN {col} TEXT NOT NULL DEFAULT ''"
                    )
            except Exception:
                pass  # column already exists — safe to ignore

    # ── Users ─────────────────────────────────────────────────────────────────

    def add_user(self, name: str) -> int:
        now = _now()
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO users (name, created_at) VALUES (?, ?)", (name, now)
            )
            return cur.lastrowid

    def get_or_create_user(self, name: str) -> int:
        row = self.get_user_by_name(name)
        return row.id if row else self.add_user(name)

    def get_user_by_name(self, name: str) -> UserRow | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT id, name, created_at FROM users WHERE name = ?", (name,)
            ).fetchone()
        return UserRow(**dict(row)) if row else None

    def list_users(self) -> list[UserRow]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT id, name, created_at FROM users ORDER BY name"
            ).fetchall()
        return [UserRow(**dict(r)) for r in rows]

    def delete_user(self, user_id: int) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM users WHERE id = ?", (user_id,))

    # ── Captures ──────────────────────────────────────────────────────────────

    def add_capture(self, user_id: int, image_path: str) -> int:
        now = _now()
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO captures (user_id, image_path, created_at) VALUES (?, ?, ?)",
                (user_id, image_path, now),
            )
            return cur.lastrowid

    def list_captures(self, user_id: int) -> list[CaptureRow]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT id, user_id, image_path, created_at "
                "FROM captures WHERE user_id = ? ORDER BY created_at",
                (user_id,),
            ).fetchall()
        return [CaptureRow(**dict(r)) for r in rows]

    def capture_count(self, user_id: int) -> int:
        with self._connect() as con:
            return con.execute(
                "SELECT COUNT(*) FROM captures WHERE user_id = ?", (user_id,)
            ).fetchone()[0]

    # ── Adversarial attacks ───────────────────────────────────────────────────

    def clear_attacks(self) -> int:
        """Delete all attack records. Returns the number of rows deleted."""
        with self._connect() as con:
            cur = con.execute("DELETE FROM attacks")
            return cur.rowcount

    def clear_all_data(self) -> dict[str, int]:
        """Delete all attack records, captures, and users. Returns counts."""
        with self._connect() as con:
            atk = con.execute("DELETE FROM attacks").rowcount
            cap = con.execute("DELETE FROM captures").rowcount
            usr = con.execute("DELETE FROM users").rowcount
        return {"attacks": atk, "captures": cap, "users": usr}

    def add_attack(
        self,
        attacker_name: str,
        model_name: str,
        similarity_score: float,
        image_path: str,
        identity_name: str = "",
        matched_image_path: str = "",
    ) -> int:
        now = _now()
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO attacks "
                "(attacker_name, model_name, similarity_score, image_path, "
                " identity_name, matched_image_path, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (attacker_name, model_name, similarity_score, image_path,
                 identity_name, matched_image_path, now),
            )
            return cur.lastrowid

    def get_ranking(
        self,
        model_name: str | None = None,
        limit: int = 50,
    ) -> list[AttackRow]:
        cols = ("id, attacker_name, model_name, similarity_score, "
                "image_path, identity_name, matched_image_path, created_at")
        with self._connect() as con:
            if model_name:
                rows = con.execute(
                    f"SELECT {cols} FROM attacks WHERE model_name = ? "
                    "ORDER BY similarity_score ASC LIMIT ?",
                    (model_name, limit),
                ).fetchall()
            else:
                rows = con.execute(
                    f"SELECT {cols} FROM attacks "
                    "ORDER BY similarity_score ASC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [AttackRow(**dict(r)) for r in rows]

    def best_attack_per_attacker(self, model_name: str) -> list[AttackRow]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT id, attacker_name, model_name, "
                "MIN(similarity_score) AS similarity_score, "
                "image_path, identity_name, matched_image_path, created_at "
                "FROM attacks WHERE model_name = ? "
                "GROUP BY attacker_name "
                "ORDER BY similarity_score ASC",
                (model_name,),
            ).fetchall()
        return [AttackRow(**dict(r)) for r in rows]


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
