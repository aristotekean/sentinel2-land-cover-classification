#!/bin/bash
# Creates the three databases of the stack on the first start of the postgres
# volume (the official entrypoint runs every script of this folder once, with
# the container environment, before the server accepts TCP connections).
#
#   MLFLOW_DB_NAME   MLflow backend store
#   AIRFLOW_DB_NAME  Airflow metadata
#   TRAIN_DB_NAME    training database, with the PostGIS extension
#
# Each database is owned by its own user; the superuser POSTGRES_USER keeps
# administration only. Table layout of the training database lives in
# ../migrations and is applied by the train-db-migrate service.
set -eu

create_database() {
    local db="$1" user="$2" password="$3"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
        -v db="$db" -v user="$user" -v password="$password" <<-'EOSQL'
        CREATE USER :"user" WITH PASSWORD :'password';
        CREATE DATABASE :"db" OWNER :"user";
EOSQL
    echo "database ready: $db (owner $user)"
}

create_database "$MLFLOW_DB_NAME" "$MLFLOW_DB_USER" "$MLFLOW_DB_PASSWORD"
create_database "$AIRFLOW_DB_NAME" "$AIRFLOW_DB_USER" "$AIRFLOW_DB_PASSWORD"
create_database "$TRAIN_DB_NAME" "$TRAIN_DB_USER" "$TRAIN_DB_PASSWORD"

# Installing an extension needs a superuser; the tables created later by the
# migrations belong to TRAIN_DB_USER.
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$TRAIN_DB_NAME" \
    -c "CREATE EXTENSION IF NOT EXISTS postgis;"
echo "postgis enabled on: $TRAIN_DB_NAME"
