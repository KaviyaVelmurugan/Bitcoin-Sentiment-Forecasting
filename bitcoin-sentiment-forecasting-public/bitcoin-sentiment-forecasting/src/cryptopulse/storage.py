"""SQLite operational state, migrations, run history, and overlap protection."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib.resources import files
from pathlib import Path
from typing import Iterator

from . import __version__


LOCK_NAME = "main_pipeline"
LOCK_MAX_AGE = timedelta(hours=2)


class PipelineBusyError(RuntimeError):
    """Raised when another non-stale pipeline run owns the operational lock."""


@dataclass(frozen=True, slots=True)
class PipelineRun:
    run_id: str
    started_at: str
    finished_at: str | None
    status: str
    current_stage: str
    error_message: str | None
    report_root: str
    application_version: str


class OperationalStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def migrate(self, *, applied_at: datetime) -> list[int]:
        if applied_at.tzinfo is None or applied_at.utcoffset() != UTC.utcoffset(applied_at):
            raise ValueError("migration timestamp must use UTC")
        applied: list[int] = []
        migration_root = files("cryptopulse").joinpath("migrations")
        with self.connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            known = {
                row["version"] for row in connection.execute("SELECT version FROM schema_migrations")
            }
            for resource in sorted(migration_root.iterdir(), key=lambda item: item.name):
                if not resource.name.endswith(".sql"):
                    continue
                version = int(resource.name.split("_", 1)[0])
                if version in known:
                    continue
                connection.executescript(resource.read_text(encoding="utf-8"))
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (version, applied_at.isoformat().replace("+00:00", "Z")),
                )
                applied.append(version)
        return applied

    def start_run(
        self,
        run_id: str,
        *,
        started_at: datetime,
        report_root: Path,
    ) -> None:
        if started_at.tzinfo is None or started_at.utcoffset() != UTC.utcoffset(started_at):
            raise ValueError("pipeline start timestamp must use UTC")
        timestamp = started_at.isoformat().replace("+00:00", "Z")
        stale_before = (started_at - LOCK_MAX_AGE).isoformat().replace("+00:00", "Z")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT owner_run_id, acquired_at FROM pipeline_locks WHERE lock_name = ?",
                (LOCK_NAME,),
            ).fetchone()
            if existing and existing["acquired_at"] >= stale_before:
                raise PipelineBusyError(f"pipeline is already running as {existing['owner_run_id']}")
            if existing:
                connection.execute(
                    "UPDATE pipeline_runs SET finished_at = ?, status = 'failed', "
                    "current_stage = 'failed', error_message = ? "
                    "WHERE run_id = ? AND status = 'running'",
                    (timestamp, "stale pipeline lock recovered", existing["owner_run_id"]),
                )
                connection.execute("DELETE FROM pipeline_locks WHERE lock_name = ?", (LOCK_NAME,))
            connection.execute(
                "INSERT INTO pipeline_locks(lock_name, owner_run_id, acquired_at) VALUES (?, ?, ?)",
                (LOCK_NAME, run_id, timestamp),
            )
            connection.execute(
                "INSERT INTO pipeline_runs(run_id, started_at, status, current_stage, report_root, "
                "application_version) VALUES (?, ?, 'running', 'starting', ?, ?)",
                (run_id, timestamp, str(report_root), __version__),
            )

    def update_stage(self, run_id: str, stage: str) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE pipeline_runs SET current_stage = ? WHERE run_id = ? AND status = 'running'",
                (stage, run_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"running pipeline record not found: {run_id}")

    def finish_run(
        self,
        run_id: str,
        *,
        finished_at: datetime,
        succeeded: bool,
        error_message: str | None = None,
    ) -> None:
        status = "succeeded" if succeeded else "failed"
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE pipeline_runs SET finished_at = ?, status = ?, current_stage = ?, "
                "error_message = ? WHERE run_id = ? AND status = 'running'",
                (
                    finished_at.isoformat().replace("+00:00", "Z"), status, "complete" if succeeded else "failed",
                    error_message, run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"running pipeline record not found: {run_id}")
            connection.execute(
                "DELETE FROM pipeline_locks WHERE lock_name = ? AND owner_run_id = ?",
                (LOCK_NAME, run_id),
            )

    def recent_runs(self, limit: int = 10) -> list[PipelineRun]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [PipelineRun(**dict(row)) for row in rows]
