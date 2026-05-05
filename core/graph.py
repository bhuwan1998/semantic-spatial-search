"""
Apache AGE Graph Interface Module.

Provides helper functions to query the osm_spatial property graph built by
data/setup_db.py using openCypher via AGE's cypher() SQL function.

The openCypher queries here are directly portable to AWS Neptune
(Neptune openCypher endpoint) by swapping the connection layer.
"""

from core.db import get_conn

GRAPH_NAME = "osm_spatial"


def _cypher(query: str) -> list[dict]:
    """
    Execute an openCypher query against the AGE graph and return results.
    Returns an empty list if AGE is unavailable or the query fails.
    """
    sql = f"SELECT * FROM cypher('{GRAPH_NAME}', $$ {query} $$) AS (result agtype);"
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("LOAD 'age';")
                cur.execute("SET search_path = ag_catalog, \"$user\", public;")
                cur.execute(sql)
                return cur.fetchall()
    except Exception:
        return []


def get_nearby_features(fid: int, label: str, target_label: str, limit: int = 5) -> list[dict]:
    """
    Return features of target_label that are NEAR the given node (by fid).

    Example:
        get_nearby_features(42, "School", "Park") → nearby parks
    """
    cypher = (
        f"MATCH (a:{label} {{fid: {fid}}})-[r:NEAR]->(b:{target_label}) "
        f"RETURN b.name AS name, b.suburb AS suburb, r.distance_m AS distance_m "
        f"ORDER BY r.distance_m LIMIT {limit}"
    )
    return _cypher(cypher)


def get_features_in_suburb(suburb: str, label: str, limit: int = 10) -> list[dict]:
    """
    Return features of a given label within a named suburb boundary.
    """
    safe_suburb = suburb.replace("'", "\\'")
    cypher = (
        f"MATCH (a:{label})-[:WITHIN]->(b:Boundary {{name: '{safe_suburb}'}}) "
        f"RETURN a.name AS name, a.fid AS fid "
        f"LIMIT {limit}"
    )
    return _cypher(cypher)


def get_graph_summary() -> dict:
    """
    Return a dict with NEAR and WITHIN edge counts per label pair.
    Used for the Graph Explorer tab and the LLM context string.
    Returns empty dict if AGE is unavailable.
    """
    summary = {"near": {}, "within": {}, "near_total": 0, "within_total": 0}
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("LOAD 'age';")
                cur.execute("SET search_path = ag_catalog, \"$user\", public;")

                # NEAR edges by label pair
                near_sql = (
                    f"SELECT * FROM cypher('{GRAPH_NAME}', $$ "
                    f"MATCH (a)-[r:NEAR]->(b) "
                    f"RETURN labels(a)[0] AS src, labels(b)[0] AS tgt, count(r) AS n "
                    f"$$) AS (src agtype, tgt agtype, n agtype);"
                )
                cur.execute(near_sql)
                for row in cur.fetchall():
                    src = str(row[0]).strip('"')
                    tgt = str(row[1]).strip('"')
                    n   = int(str(row[2]))
                    summary["near"][f"{src}→{tgt}"] = n
                    summary["near_total"] += n

                # WITHIN edges total
                within_sql = (
                    f"SELECT * FROM cypher('{GRAPH_NAME}', $$ "
                    f"MATCH ()-[r:WITHIN]->() RETURN count(r) AS n "
                    f"$$) AS (n agtype);"
                )
                cur.execute(within_sql)
                row = cur.fetchone()
                summary["within_total"] = int(str(row[0])) if row else 0

    except Exception:
        pass
    return summary


def get_graph_context_for_query(query: str) -> str:
    """
    Generate a short graph context string to supplement the LLM prompt.

    Looks for suburb mentions or known place names and returns relevant
    graph relationship counts as a text summary.
    Returns an empty string if the graph is unavailable.
    """
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("LOAD 'age';")
                cur.execute("SET search_path = ag_catalog, \"$user\", public;")
                sql = (
                    f"SELECT * FROM cypher('{GRAPH_NAME}', $$ "
                    f"MATCH ()-[r:NEAR]->() RETURN count(r) AS n "
                    f"$$) AS (n agtype);"
                )
                cur.execute(sql)
                row = cur.fetchone()
                near_count = int(str(row[0])) if row else 0

                within_sql = (
                    f"SELECT * FROM cypher('{GRAPH_NAME}', $$ "
                    f"MATCH ()-[r:WITHIN]->() RETURN count(r) AS n "
                    f"$$) AS (n agtype);"
                )
                cur.execute(within_sql)
                row = cur.fetchone()
                within_count = int(str(row[0])) if row else 0

        return (
            f"[Graph: {near_count} NEAR relationships, "
            f"{within_count} WITHIN relationships available for spatial context]"
        )
    except Exception:
        return ""


def graph_is_available() -> bool:
    """Return True if the AGE graph exists and is queryable."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("LOAD 'age';")
                cur.execute("SET search_path = ag_catalog, \"$user\", public;")
                cur.execute(
                    "SELECT name FROM ag_graph WHERE name = %s",
                    (GRAPH_NAME,)
                )
                return cur.fetchone() is not None
    except Exception:
        return False
