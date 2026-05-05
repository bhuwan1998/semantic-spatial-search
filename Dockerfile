# Dockerfile — PostgreSQL 15 + Apache AGE + pgvector + PostGIS
#
# Apache AGE only supports PG 11-16; official image uses PG 15.
# We compile pgvector from source into the AGE base image,
# then install PostGIS via apt.
#
# Build: docker build -t geospatial-db .
# Run via docker-compose.yml (preferred).

FROM apache/age:release_PG15_1.6.0

# Switch to root to install packages
USER root

# Install build dependencies and PostGIS
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        postgresql-server-dev-15 \
        postgresql-15-postgis-3 \
        postgresql-15-postgis-3-scripts \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Compile and install pgvector from source (v0.8.0)
# OPTFLAGS="" disables -march=native which causes SIGSEGV inside Docker's Linux VM
RUN git clone --branch v0.8.0 --depth 1 https://github.com/pgvector/pgvector.git /tmp/pgvector \
    && cd /tmp/pgvector \
    && make OPTFLAGS="" \
    && make install \
    && rm -rf /tmp/pgvector

# Copy initialization SQL that loads extensions and creates base schema
COPY docker/initdb.sql /docker-entrypoint-initdb.d/01_initdb.sql

# Drop back to postgres user (AGE image default)
USER postgres
