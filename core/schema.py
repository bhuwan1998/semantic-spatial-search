"""
Schema Introspection Module - Reads table/column/geometry metadata from PostgreSQL + PostGIS.

Provides the LLM with structured schema context so it generates correct SQL
referencing actual table and column names.
"""

from dataclasses import dataclass, field

import psycopg

from core.db import get_conn


@dataclass
class ColumnInfo:
    name: str
    data_type: str
    is_geometry: bool = False
    geometry_type: str | None = None
    srid: int | None = None


@dataclass
class TableInfo:
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    row_count: int = 0
    geometry_column: str | None = None
    geometry_type: str | None = None
    srid: int | None = None
    sample_values: dict[str, list[str]] = field(default_factory=dict)


def introspect_db() -> list[TableInfo]:
    """
    Read all OSM spatial tables from PostgreSQL, including column info,
    geometry metadata from geometry_columns, row counts, and sample values.

    Reads from both 'public' and 'ag_catalog' schemas so that osm_boundaries
    and osm_all (which live in ag_catalog due to AGE setup) are always included.
    """
    with get_conn() as conn:
        # ----- geometry_columns metadata (both schemas) -----
        geom_info: dict[str, dict] = {}
        with conn.cursor() as cur:
            cur.execute("""
                SELECT f_table_name, f_geometry_column, type, srid
                FROM geometry_columns
                WHERE f_table_schema IN ('public', 'ag_catalog')
            """)
            for row in cur.fetchall():
                geom_info[row["f_table_name"]] = {
                    "column": row["f_geometry_column"],
                    "type":   row["type"],
                    "srid":   row["srid"],
                }

        # ----- list our OSM tables from both public and ag_catalog -----
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name, table_schema
                FROM information_schema.tables
                WHERE table_schema IN ('public', 'ag_catalog')
                  AND (table_name LIKE 'osm_%')
                  AND table_type IN ('BASE TABLE', 'VIEW')
                ORDER BY table_name
            """)
            table_rows = [(r["table_name"], r["table_schema"]) for r in cur.fetchall()]

        tables: list[TableInfo] = []

        for table_name, table_schema in table_rows:
            table = TableInfo(name=table_name)

            # ----- columns -----
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name, data_type, udt_name
                    FROM information_schema.columns
                    WHERE table_schema = %s
                      AND table_name = %s
                    ORDER BY ordinal_position
                """, (table_schema, table_name))
                for row in cur.fetchall():
                    col_name  = row["column_name"]
                    col_type  = row["udt_name"] if row["data_type"] == "USER-DEFINED" else row["data_type"]
                    is_geom   = False
                    geom_type = None
                    srid      = None

                    if table_name in geom_info and col_name == geom_info[table_name]["column"]:
                        is_geom   = True
                        geom_type = geom_info[table_name]["type"]
                        srid      = geom_info[table_name]["srid"]
                        table.geometry_column = col_name
                        table.geometry_type   = geom_type
                        table.srid            = srid

                    table.columns.append(ColumnInfo(
                        name=col_name,
                        data_type="GEOMETRY" if is_geom else col_type,
                        is_geometry=is_geom,
                        geometry_type=geom_type,
                        srid=srid,
                    ))

            # ----- row count -----
            try:
                with conn.cursor() as cur:
                    cur.execute(f'SELECT COUNT(*) AS n FROM {table_schema}."{table_name}"')
                    table.row_count = cur.fetchone()["n"]
            except psycopg.Error:
                pass

            # ----- sample values for text / categorical columns -----
            for col in table.columns:
                if col.is_geometry or col.name in ("id", "osm_id"):
                    continue
                if col.data_type not in ("text", "character varying", "varchar"):
                    continue
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            f'SELECT DISTINCT "{col.name}" '
                            f'FROM {table_schema}."{table_name}" '
                            f'WHERE "{col.name}" IS NOT NULL '
                            f'LIMIT 8'
                        )
                        values = [str(r[col.name]) for r in cur.fetchall() if r[col.name]]
                        if values:
                            table.sample_values[col.name] = values
                except psycopg.Error:
                    pass

            tables.append(table)

    return tables


def format_schema_for_llm(tables: list[TableInfo]) -> str:
    """Format the schema as a concise string for LLM prompt context."""
    lines = []
    for table in tables:
        geom_desc = ""
        if table.geometry_type and table.srid:
            geom_desc = f" (geometry: {table.geometry_type}, SRID={table.srid})"
        lines.append(f"TABLE {table.name} ({table.row_count} rows){geom_desc}:")

        for col in table.columns:
            if col.is_geometry:
                lines.append(f"  - {col.name}: {col.geometry_type} (SRID={col.srid})")
            else:
                sample = ""
                if col.name in table.sample_values:
                    vals = table.sample_values[col.name][:5]
                    sample = f"  -- e.g. {', '.join(repr(v) for v in vals)}"
                lines.append(f"  - {col.name}: {col.data_type}{sample}")

        lines.append("")

    return "\n".join(lines)


def get_table_names(tables: list[TableInfo]) -> set[str]:
    """Extract just the table names as a set."""
    return {t.name for t in tables}
