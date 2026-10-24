#!/usr/bin/env bash
# Make sure Docker is running and Postgres and Qdrant are up (macOS: opens Docker Desktop).
# Safe to run any time; does nothing when everything is already up. Langfuse is not started.
set -euo pipefail
cd "$(dirname "$0")/.."

WAIT_SECONDS=${WAIT_SECONDS:-120}

wait_for() {  # wait_for "what" command...
  local what=$1
  shift
  for ((i = 0; i < WAIT_SECONDS; i++)); do
    if "$@" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  echo "ensure_docker: timed out after ${WAIT_SECONDS}s waiting for $what" >&2
  return 1
}

if ! docker info >/dev/null 2>&1; then
  if [[ "$(uname)" != "Darwin" ]]; then
    echo "ensure_docker: Docker is not running; start it and run this again" >&2
    exit 1
  fi
  echo "ensure_docker: starting Docker Desktop…"
  open -a Docker
  wait_for "the Docker daemon" docker info
fi

docker compose up -d postgres qdrant
wait_for "Postgres on port 5433" docker compose exec -T postgres pg_isready -U workout
echo "ensure_docker: Docker, Postgres and Qdrant are up"
