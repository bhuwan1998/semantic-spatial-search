-- docker/initdb.sql
-- Bootstraps the geospatial database with PostGIS, pgvector, and Apache AGE extensions.
-- Runs automatically on first container start (placed in /docker-entrypoint-initdb.d/).

-- Create the application database (the initdb script runs as the superuser defined
-- by POSTGRES_USER, inside the database defined by POSTGRES_DB).

-- Load extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS vector;
LOAD 'age';
CREATE EXTENSION IF NOT EXISTS age;

-- Make AGE catalog accessible by default
ALTER DATABASE geospatial SET search_path = ag_catalog, "$user", public;

-- pgvector table for hybrid RAG schema retrieval
CREATE TABLE IF NOT EXISTS table_descriptions (
    id          serial PRIMARY KEY,
    table_name  text NOT NULL UNIQUE,
    description text NOT NULL,
    embedding   vector(768)
);

-- Spatial tables (created here so setup_db.py can use INSERT ... ON CONFLICT)
CREATE TABLE IF NOT EXISTS osm_schools (
    id          bigserial PRIMARY KEY,
    osm_id      bigint,
    name        text,
    amenity     text,
    "addr:street"      text,
    "addr:housenumber" text,
    operator    text,
    opening_hours text,
    geometry    geometry(Geometry, 4326)
);
CREATE INDEX IF NOT EXISTS osm_schools_geom_idx ON osm_schools USING GIST (geometry);

CREATE TABLE IF NOT EXISTS osm_hospitals (
    id          bigserial PRIMARY KEY,
    osm_id      bigint,
    name        text,
    amenity     text,
    "addr:street"      text,
    "addr:housenumber" text,
    operator    text,
    opening_hours text,
    geometry    geometry(Geometry, 4326)
);
CREATE INDEX IF NOT EXISTS osm_hospitals_geom_idx ON osm_hospitals USING GIST (geometry);

CREATE TABLE IF NOT EXISTS osm_restaurants (
    id          bigserial PRIMARY KEY,
    osm_id      bigint,
    name        text,
    amenity     text,
    cuisine     text,
    "addr:street"      text,
    "addr:housenumber" text,
    opening_hours text,
    geometry    geometry(Geometry, 4326)
);
CREATE INDEX IF NOT EXISTS osm_restaurants_geom_idx ON osm_restaurants USING GIST (geometry);

CREATE TABLE IF NOT EXISTS osm_pharmacies (
    id          bigserial PRIMARY KEY,
    osm_id      bigint,
    name        text,
    amenity     text,
    "addr:street"      text,
    "addr:housenumber" text,
    operator    text,
    opening_hours text,
    geometry    geometry(Geometry, 4326)
);
CREATE INDEX IF NOT EXISTS osm_pharmacies_geom_idx ON osm_pharmacies USING GIST (geometry);

CREATE TABLE IF NOT EXISTS osm_roads (
    id          bigserial PRIMARY KEY,
    osm_id      bigint,
    name        text,
    highway     text,
    surface     text,
    lanes       text,
    maxspeed    text,
    oneway      text,
    geometry    geometry(Geometry, 4326)
);
CREATE INDEX IF NOT EXISTS osm_roads_geom_idx ON osm_roads USING GIST (geometry);

CREATE TABLE IF NOT EXISTS osm_waterways (
    id          bigserial PRIMARY KEY,
    osm_id      bigint,
    name        text,
    waterway    text,
    geometry    geometry(Geometry, 4326)
);
CREATE INDEX IF NOT EXISTS osm_waterways_geom_idx ON osm_waterways USING GIST (geometry);

CREATE TABLE IF NOT EXISTS osm_railways (
    id          bigserial PRIMARY KEY,
    osm_id      bigint,
    name        text,
    railway     text,
    geometry    geometry(Geometry, 4326)
);
CREATE INDEX IF NOT EXISTS osm_railways_geom_idx ON osm_railways USING GIST (geometry);

CREATE TABLE IF NOT EXISTS osm_parks (
    id          bigserial PRIMARY KEY,
    osm_id      bigint,
    name        text,
    leisure     text,
    geometry    geometry(Geometry, 4326)
);
CREATE INDEX IF NOT EXISTS osm_parks_geom_idx ON osm_parks USING GIST (geometry);

CREATE TABLE IF NOT EXISTS osm_buildings (
    id          bigserial PRIMARY KEY,
    osm_id      bigint,
    name        text,
    building    text,
    "addr:street"      text,
    "addr:housenumber" text,
    geometry    geometry(Geometry, 4326)
);
CREATE INDEX IF NOT EXISTS osm_buildings_geom_idx ON osm_buildings USING GIST (geometry);

CREATE TABLE IF NOT EXISTS osm_landuse (
    id          bigserial PRIMARY KEY,
    osm_id      bigint,
    name        text,
    landuse     text,
    geometry    geometry(Geometry, 4326)
);
CREATE INDEX IF NOT EXISTS osm_landuse_geom_idx ON osm_landuse USING GIST (geometry);

CREATE TABLE IF NOT EXISTS osm_natural (
    id          bigserial PRIMARY KEY,
    osm_id      bigint,
    name        text,
    "natural"   text,
    geometry    geometry(Geometry, 4326)
);
CREATE INDEX IF NOT EXISTS osm_natural_geom_idx ON osm_natural USING GIST (geometry);

CREATE TABLE IF NOT EXISTS osm_boundaries (
    id          bigserial PRIMARY KEY,
    osm_id      bigint,
    name        text,
    boundary    text,
    admin_level text,
    geometry    geometry(Geometry, 4326)
);
CREATE INDEX IF NOT EXISTS osm_boundaries_geom_idx ON osm_boundaries USING GIST (geometry);

-- Unified view across all layers
CREATE OR REPLACE VIEW osm_all AS
    SELECT id, osm_id, name, 'schools'     AS layer, geometry FROM osm_schools
    UNION ALL
    SELECT id, osm_id, name, 'hospitals'   AS layer, geometry FROM osm_hospitals
    UNION ALL
    SELECT id, osm_id, name, 'restaurants' AS layer, geometry FROM osm_restaurants
    UNION ALL
    SELECT id, osm_id, name, 'pharmacies'  AS layer, geometry FROM osm_pharmacies
    UNION ALL
    SELECT id, osm_id, name, 'roads'       AS layer, geometry FROM osm_roads
    UNION ALL
    SELECT id, osm_id, name, 'waterways'   AS layer, geometry FROM osm_waterways
    UNION ALL
    SELECT id, osm_id, name, 'railways'    AS layer, geometry FROM osm_railways
    UNION ALL
    SELECT id, osm_id, name, 'parks'       AS layer, geometry FROM osm_parks
    UNION ALL
    SELECT id, osm_id, name, 'buildings'   AS layer, geometry FROM osm_buildings
    UNION ALL
    SELECT id, osm_id, name, 'landuse'     AS layer, geometry FROM osm_landuse
    UNION ALL
    SELECT id, osm_id, name, 'natural'     AS layer, geometry FROM osm_natural
    UNION ALL
    SELECT id, osm_id, name, 'boundaries'  AS layer, geometry FROM osm_boundaries;

-- ─── Performance indexes ──────────────────────────────────────────────────────
-- pg_trgm trigram indexes for fast ILIKE / similarity name lookups
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE INDEX IF NOT EXISTS osm_schools_name_trgm     ON osm_schools     USING GIN (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS osm_hospitals_name_trgm   ON osm_hospitals   USING GIN (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS osm_restaurants_name_trgm ON osm_restaurants USING GIN (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS osm_pharmacies_name_trgm  ON osm_pharmacies  USING GIN (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS osm_roads_name_trgm       ON osm_roads       USING GIN (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS osm_waterways_name_trgm   ON osm_waterways   USING GIN (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS osm_railways_name_trgm    ON osm_railways    USING GIN (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS osm_parks_name_trgm       ON osm_parks       USING GIN (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS osm_buildings_name_trgm   ON osm_buildings   USING GIN (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS osm_landuse_name_trgm     ON osm_landuse     USING GIN (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS osm_natural_name_trgm     ON osm_natural     USING GIN (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS osm_boundaries_name_trgm  ON osm_boundaries  USING GIN (name gin_trgm_ops);

-- Partial GiST index on osm_boundaries admin_level for fast suburb lookups
CREATE INDEX IF NOT EXISTS osm_boundaries_admin_geom_idx
    ON osm_boundaries USING GIST (geometry)
    WHERE admin_level IN ('8', '9', '10');

-- Materialized view: pre-unions osm_all for fast cross-layer name lookups
-- (refresh daily or after data reloads with: REFRESH MATERIALIZED VIEW CONCURRENTLY osm_all_mat;)
CREATE MATERIALIZED VIEW IF NOT EXISTS osm_all_mat AS
    SELECT id, osm_id, name, 'schools'     AS layer, geometry FROM osm_schools
    UNION ALL
    SELECT id, osm_id, name, 'hospitals'   AS layer, geometry FROM osm_hospitals
    UNION ALL
    SELECT id, osm_id, name, 'restaurants' AS layer, geometry FROM osm_restaurants
    UNION ALL
    SELECT id, osm_id, name, 'pharmacies'  AS layer, geometry FROM osm_pharmacies
    UNION ALL
    SELECT id, osm_id, name, 'roads'       AS layer, geometry FROM osm_roads
    UNION ALL
    SELECT id, osm_id, name, 'waterways'   AS layer, geometry FROM osm_waterways
    UNION ALL
    SELECT id, osm_id, name, 'railways'    AS layer, geometry FROM osm_railways
    UNION ALL
    SELECT id, osm_id, name, 'parks'       AS layer, geometry FROM osm_parks
    UNION ALL
    SELECT id, osm_id, name, 'buildings'   AS layer, geometry FROM osm_buildings
    UNION ALL
    SELECT id, osm_id, name, 'landuse'     AS layer, geometry FROM osm_landuse
    UNION ALL
    SELECT id, osm_id, name, 'natural'     AS layer, geometry FROM osm_natural
    UNION ALL
    SELECT id, osm_id, name, 'boundaries'  AS layer, geometry FROM osm_boundaries;

CREATE UNIQUE INDEX IF NOT EXISTS osm_all_mat_id_layer_idx ON osm_all_mat (id, layer);
CREATE INDEX        IF NOT EXISTS osm_all_mat_geom_idx     ON osm_all_mat USING GIST (geometry);
CREATE INDEX        IF NOT EXISTS osm_all_mat_name_trgm    ON osm_all_mat USING GIN  (name gin_trgm_ops);
