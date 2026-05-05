"""
SQL Validator Module - Validates and sanitizes LLM-generated spatial SQL.

Ensures only safe SELECT queries run against PostGIS by:
- Rejecting DML/DDL statements
- Whitelisting allowed tables and PostGIS functions
- Auto-injecting LIMIT clauses
- Stripping markdown artifacts
"""

import re

import sqlparse


# PostGIS functions that are safe to use
ALLOWED_FUNCTIONS = {
    # Spatial relationships
    "st_intersects", "st_contains", "st_within", "st_covers",
    "st_coveredby", "st_crosses", "st_touches", "st_overlaps",
    "st_disjoint", "st_equals", "st_dwithin",
    # Measurements (PostGIS)
    "st_distance", "st_area", "st_length", "st_perimeter",
    # Constructors (PostGIS)
    "st_makepoint", "st_point", "st_geomfromtext", "st_geomfromgeojson",
    "st_setsrid", "st_makeenvelope", "st_makeline", "st_makepolygon",
    "st_geogfromtext",
    # Processing
    "st_centroid", "st_union", "st_intersection", "st_difference",
    "st_convexhull", "st_transform", "st_simplify", "st_buffer",
    "st_envelope", "st_collect", "st_pointonsurface", "st_snap",
    "st_force2d", "st_multi",
    # Output
    "st_x", "st_y", "st_astext", "st_asgeojson", "st_asbinary",
    "st_srid", "st_geometrytype", "st_numgeometries", "st_numpoints",
    "st_isvalid", "st_isempty", "st_dimension",
    "geometrytype",
    # Aggregate spatial (st_extent excluded — returns SRID 0, breaks spatial ops)
    "st_collect", "st_union",
    # Cast helpers
    "geography", "geometry",
    # Standard SQL / PostgreSQL functions
    "count", "sum", "avg", "min", "max", "round", "coalesce",
    "lower", "upper", "trim", "cast", "nullif", "abs",
    "concat", "length", "substr", "substring", "replace",
    "to_char", "to_number", "greatest", "least",
    "array_agg", "string_agg", "json_agg", "jsonb_agg",
}

# Table names the LLM sometimes hallucinates — never valid in our schema
HALLUCINATED_TABLES = {
    "boundary", "boundaries", "suburb", "suburbs", "area", "areas",
    "place", "places", "location", "locations", "region", "regions",
    "feature", "features", "layer", "layers",
}
FORBIDDEN_PATTERNS = [
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b",
    r"\b(EXECUTE|EXEC)\b",
    r"\bCOPY\b",
    r"\bpg_read_file\b",
    r"\bpg_ls_dir\b",
    r";\s*\S",  # Multiple statements (statement stacking)
    r"\bST_Extent\b",  # Returns SRID 0 — use osm_boundaries subquery instead
]


class ValidationError(Exception):
    """Raised when SQL validation fails."""
    pass


class SQLValidator:
    """Validates and sanitizes LLM-generated spatial SQL."""

    def __init__(self, allowed_tables: set[str]):
        self.allowed_tables = {t.lower() for t in allowed_tables}

    def validate(self, sql: str) -> str:
        """
        Validate and clean SQL. Returns cleaned SQL or raises ValidationError.
        """
        # Step 1: Strip markdown artifacts
        sql = self._strip_markdown(sql)

        if not sql.strip():
            raise ValidationError("Empty SQL query")

        # Step 2: Check forbidden patterns
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, sql, re.IGNORECASE):
                raise ValidationError(
                    f"Forbidden SQL pattern detected: {pattern}"
                )

        # Step 3: Parse and validate structure
        try:
            parsed = sqlparse.parse(sql)
        except Exception as e:
            raise ValidationError(f"SQL parse error: {e}")

        if len(parsed) == 0:
            raise ValidationError("No SQL statement found")

        if len(parsed) > 1:
            raise ValidationError(
                "Only single SQL statements are allowed. "
                "Found multiple statements separated by semicolons."
            )

        stmt = parsed[0]
        stmt_type = stmt.get_type()
        # sqlparse returns None for CTE queries (WITH ... SELECT) — treat as SELECT
        if stmt_type not in ("SELECT", None):
            raise ValidationError(
                f"Only SELECT statements are allowed, got: {stmt_type}"
            )

        # Step 4: Validate table references
        tables_used = self._extract_tables(sql)
        hallucinated = tables_used & HALLUCINATED_TABLES
        if hallucinated:
            raise ValidationError(
                f"Hallucinated table name(s) used: {hallucinated}. "
                f"Use only osm_* tables. For suburb boundaries use osm_boundaries. "
                f"Never use a CTE name as if it were a real table in FROM/JOIN outside the CTE body."
            )
        unknown_tables = tables_used - self.allowed_tables
        if unknown_tables:
            raise ValidationError(
                f"Unknown table(s) referenced: {unknown_tables}. "
                f"Available tables: {self.allowed_tables}"
            )

        # Step 5: Validate function names
        functions_used = self._extract_functions(sql)
        unknown_funcs = functions_used - ALLOWED_FUNCTIONS
        if unknown_funcs:
            raise ValidationError(
                f"Unknown or disallowed function(s): {unknown_funcs}. "
                f"Use only PostGIS spatial functions."
            )

        # Step 6: Expand ORDER BY alias references to full expressions
        sql = self._expand_order_by_aliases(sql)

        # Step 7: Quote column names containing colons (e.g. addr:street)
        sql = self._quote_colon_columns(sql)

        # Step 8: Ensure LIMIT exists
        if not re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
            sql = sql.rstrip().rstrip(";") + " LIMIT 500;"

        # Step 9: Ensure trailing semicolon
        sql = sql.rstrip()
        if not sql.endswith(";"):
            sql += ";"

        return sql

    def _strip_markdown(self, sql: str) -> str:
        """Remove markdown code fences and language hints."""
        sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.MULTILINE)
        sql = re.sub(r"```\s*$", "", sql, flags=re.MULTILINE)
        # Strip leading/trailing explanation text — find where the SQL starts.
        # Must match WITH (CTEs) before SELECT, otherwise the WITH clause gets
        # stripped and CTE queries arrive truncated (missing "WITH name AS (").
        match = re.search(r"(\bWITH\b.+|\bSELECT\b.+)", sql, re.IGNORECASE | re.DOTALL)
        if match:
            sql = match.group(1)
        return sql.strip()

    def _extract_tables(self, sql: str) -> set[str]:
        """Extract table names from FROM and JOIN clauses, excluding CTE names."""
        # Collect CTE names first (WITH cte_name AS (...)) so we don't flag them
        cte_names = {
            m.lower()
            for m in re.findall(r'\bWITH\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s*\(', sql, re.IGNORECASE)
        }
        # Also catch subsequent CTE definitions: , cte_name AS (
        cte_names |= {
            m.lower()
            for m in re.findall(r',\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s*\(', sql, re.IGNORECASE)
        }

        # Match FROM table and JOIN table (with optional alias)
        pattern = r"(?:FROM|JOIN)\s+\"?([a-zA-Z_][a-zA-Z0-9_]*)\"?"
        matches = re.findall(pattern, sql, re.IGNORECASE)
        # Filter out SQL keywords and CTE names
        skip = {"select", "where", "on", "and", "or", "not", "lateral"}
        return {
            m.lower() for m in matches
            if m.lower() not in skip and m.lower() not in cte_names
        }

    def _extract_functions(self, sql: str) -> set[str]:
        """Extract function call names from SQL."""
        pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
        matches = re.findall(pattern, sql, re.IGNORECASE)
        # Filter out SQL keywords that syntactically look like functions
        keywords = {
            "select", "from", "where", "and", "or", "not", "in",
            "case", "when", "then", "else", "end", "as", "on",
            "join", "left", "right", "inner", "outer", "cross",
            "group", "order", "having", "limit", "offset", "exists",
            "between", "like", "ilike", "glob", "values", "over",
            "partition", "row", "rows", "range", "preceding", "following",
        }
        return {m.lower() for m in matches if m.lower() not in keywords}

    def _expand_order_by_aliases(self, sql: str) -> str:
        """
        Replace alias references in ORDER BY with the full expression.

        The LLM sometimes generates queries like:
            SELECT ST_Distance(...) * 111320 AS distance_meters ... ORDER BY distance_meters
        but SQLite may reject the alias in ORDER BY in some contexts.
        This method extracts `expression AS alias` mappings from the SELECT clause
        and replaces any alias references found in ORDER BY with the original expression.
        """
        # Find the ORDER BY clause
        order_match = re.search(r'\bORDER\s+BY\b(.+?)(?:\bLIMIT\b|;|$)', sql, re.IGNORECASE | re.DOTALL)
        if not order_match:
            return sql

        # Extract SELECT ... FROM portion to find aliases
        select_match = re.search(r'\bSELECT\b(.+?)\bFROM\b', sql, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return sql

        select_body = select_match.group(1)

        # Parse out `expression AS alias` mappings.
        # We need to handle nested parentheses in expressions like ST_Distance(...) * 111320 AS distance_meters.
        # Strategy: split SELECT columns by commas at depth 0, then check for AS alias.
        alias_map = {}
        depth = 0
        current_token = []
        for char in select_body:
            if char == '(':
                depth += 1
                current_token.append(char)
            elif char == ')':
                depth -= 1
                current_token.append(char)
            elif char == ',' and depth == 0:
                col_expr = ''.join(current_token).strip()
                self._extract_alias(col_expr, alias_map)
                current_token = []
            else:
                current_token.append(char)
        # Handle the last column
        col_expr = ''.join(current_token).strip()
        self._extract_alias(col_expr, alias_map)

        if not alias_map:
            return sql

        # Replace alias references in the ORDER BY clause
        order_text = order_match.group(1)
        new_order_text = order_text

        for alias, expression in alias_map.items():
            # Replace standalone alias references (word boundary match, case-insensitive)
            new_order_text = re.sub(
                r'\b' + re.escape(alias) + r'\b',
                expression,
                new_order_text,
                flags=re.IGNORECASE,
            )

        if new_order_text != order_text:
            sql = sql[:order_match.start(1)] + new_order_text + sql[order_match.end(1):]

        return sql

    def _extract_alias(self, col_expr: str, alias_map: dict) -> None:
        """Extract an AS alias mapping from a single SELECT column expression."""
        # Match `<expression> AS <alias>` at the end.
        # The alias is the last word after AS (case-insensitive).
        # We use a regex that captures everything before AS and the alias after.
        match = re.match(
            r'^(.+?)\s+[Aa][Ss]\s+\"?([a-zA-Z_][a-zA-Z0-9_]*)\"?\s*$',
            col_expr,
            re.DOTALL,
        )
        if match:
            expression = match.group(1).strip()
            alias = match.group(2).strip()
            alias_map[alias] = expression

    def _quote_colon_columns(self, sql: str) -> str:
        """
        Auto-quote column names containing colons (e.g. addr:street -> "addr:street").

        OSM data has columns like addr:street and addr:housenumber. SQLite requires
        these to be double-quoted because the colon is not a valid identifier character.
        This method finds unquoted word:word patterns and wraps them in double quotes,
        being careful not to double-quote already-quoted references.
        """
        # Match unquoted identifier:identifier patterns that are NOT already
        # inside double quotes or single quotes.
        # Pattern: a word boundary, then word:word (possibly word:word:word), not preceded by "
        sql = re.sub(
            r'(?<!")(\b[a-zA-Z_]\w*(?::[a-zA-Z_]\w*)+)(?!")',
            r'"\1"',
            sql,
        )
        return sql
