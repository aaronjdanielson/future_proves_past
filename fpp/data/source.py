"""Read-only upstream SQLite access with an explicit table allowlist.

The default audit reads schemas, not player outcomes. Inventory counts are
optional and are never presented as date-eligible training/test sample sizes.
"""
import hashlib
import json
from pathlib import Path
import sqlite3

UPSTREAM_TABLES = frozenset({"players", "team_rosters", "ncaa_gamelogs", "intl_gamelogs",
                            "ncaa_summaries", "intl_summaries", "teams", "league_catalog",
                            "intl_team_boxscores", "ncaa_team_boxscores"})
CONTRACT_COLUMNS = {
    "intl_gamelogs": {"date", "home", "starter", "season", "min"},
    "league_catalog": {"league", "country", "tier", "regulation_min", "ot_min"},
    "intl_team_boxscores": {"date"},
}


class ReadOnlyRoster:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve(strict=True)
        if not self.path.is_file():
            raise ValueError("Database path must identify an existing file")
        self.connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        self.connection.execute("PRAGMA query_only = ON")
        self.connection.row_factory = sqlite3.Row

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.connection.close()

    def schema(self):
        tables = self.connection.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        return {row["name"]: {"sql": row["sql"], "columns": self.columns(row["name"])}
                for row in tables if row["name"] in UPSTREAM_TABLES}

    def columns(self, table):
        self._check_table(table)
        return [dict(row) for row in self.connection.execute(f'PRAGMA table_info("{table}")')]

    def count(self, table):
        self._check_table(table)
        return self.connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]

    @staticmethod
    def _check_table(table):
        if table not in UPSTREAM_TABLES:
            raise ValueError(f"Table is not allowlisted: {table!r}")


def audit_database(path, *, include_counts=False):
    """Schema audit only by default; no outcome rows or holdout evaluation."""
    with ReadOnlyRoster(path) as source:
        schema = source.schema()
        stat = source.path.stat()
        contracts = {}
        for table, expected in CONTRACT_COLUMNS.items():
            columns = {c["name"] for c in schema.get(table, {}).get("columns", [])}
            contracts[table] = {"table_present": table in schema,
                                "missing_columns": sorted(expected - columns)}
        result = {
            "path": str(source.path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "read_only": True, "audit_scope": "schema_and_counts" if include_counts else "schema_only",
            "tables": schema, "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
            "contract_checks": contracts,
            "temporal_eligibility_verified": False,
            "outcome_completeness_verified": False,
            "minutes_provenance_verified": False,
            "counts_are_eligibility_counts": False,
        }
        if include_counts:
            result["table_row_counts"] = {table: source.count(table) for table in schema}
        return result
