"""
Schema Introspection Module - Reads table/column/geometry metadata from a GeoPackage.

Provides the LLM with structured schema context so it generates correct SQL
referencing actual table and column names.
"""

import sqlite3
from dataclasses import dataclass, field


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


def get_spatialite_connection(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection with SpatiaLite loaded."""
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    search_paths = [
        "mod_spatialite",
        "/opt/homebrew/lib/mod_spatialite",
        "/usr/local/lib/mod_spatialite",
    ]
    for path in search_paths:
        try:
            conn.load_extension(path)
            return conn
        except sqlite3.OperationalError:
            continue
    raise RuntimeError(
        "mod_spatialite not found. Install with: brew install libspatialite"
    )


def introspect_gpkg(db_path: str) -> list[TableInfo]:
    """Read all user tables, columns, geometry info, and sample values from a GeoPackage."""
    conn = get_spatialite_connection(db_path)

    # Get geometry columns from gpkg_geometry_columns
    geom_info = {}
    try:
        rows = conn.execute("""
            SELECT table_name, column_name, geometry_type_name, srs_id
            FROM gpkg_geometry_columns
        """).fetchall()
        for table_name, col_name, geom_type, srs_id in rows:
            geom_info[table_name] = {
                "column": col_name,
                "type": geom_type,
                "srid": srs_id,
            }
    except sqlite3.OperationalError:
        pass  # Not a GPKG or table doesn't exist

    # Get user tables from gpkg_contents
    table_names = []
    try:
        rows = conn.execute("""
            SELECT table_name FROM gpkg_contents
            WHERE data_type = 'features'
            ORDER BY table_name
        """).fetchall()
        table_names = [r[0] for r in rows]
    except sqlite3.OperationalError:
        # Fallback: get all non-system tables
        rows = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type = 'table'
            AND name NOT LIKE 'gpkg_%'
            AND name NOT LIKE 'rtree_%'
            AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """).fetchall()
        table_names = [r[0] for r in rows]

    tables = []
    for table_name in table_names:
        table = TableInfo(name=table_name)

        # Get column info via PRAGMA
        pragma_rows = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        for _, col_name, col_type, _, _, _ in pragma_rows:
            is_geom = False
            geom_type = None
            srid = None

            if table_name in geom_info and col_name == geom_info[table_name]["column"]:
                is_geom = True
                geom_type = geom_info[table_name]["type"]
                srid = geom_info[table_name]["srid"]
                table.geometry_column = col_name
                table.geometry_type = geom_type
                table.srid = srid

            table.columns.append(ColumnInfo(
                name=col_name,
                data_type=col_type if not is_geom else "GEOMETRY",
                is_geometry=is_geom,
                geometry_type=geom_type,
                srid=srid,
            ))

        # Get row count
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM \"{table_name}\"").fetchone()[0]
            table.row_count = count
        except sqlite3.OperationalError:
            pass

        # Get sample values for text columns (useful for LLM context)
        for col in table.columns:
            if col.is_geometry or col.name in ("fid", "ogc_fid"):
                continue
            try:
                sample_rows = conn.execute(
                    f"SELECT DISTINCT \"{col.name}\" FROM \"{table_name}\" "
                    f"WHERE \"{col.name}\" IS NOT NULL "
                    f"LIMIT 8"
                ).fetchall()
                values = [str(r[0]) for r in sample_rows if r[0] is not None]
                if values:
                    table.sample_values[col.name] = values
            except sqlite3.OperationalError:
                pass

        tables.append(table)

    conn.close()
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
