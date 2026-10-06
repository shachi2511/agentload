#!/usr/bin/env bash
# Phase 6 (native, no Docker): start Prometheus + Grafana in the background for this repo.
# Data and logs go to .observability/ (gitignored). Stop with: ./scripts/obs_down.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
STATE="$ROOT/.observability"
mkdir -p "$STATE/prometheus-data" "$STATE/grafana-data" "$STATE/logs"

command -v prometheus >/dev/null || { echo "prometheus not found: brew install prometheus"; exit 1; }
command -v grafana >/dev/null || { echo "grafana not found: brew install grafana"; exit 1; }

running() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

if running "$STATE/prometheus.pid"; then
  echo "prometheus already running (pid $(cat "$STATE/prometheus.pid"))"
else
  prometheus --config.file="$ROOT/prometheus/prometheus.local.yml" \
    --storage.tsdb.path="$STATE/prometheus-data" \
    --web.listen-address=127.0.0.1:9090 >"$STATE/logs/prometheus.log" 2>&1 &
  echo $! >"$STATE/prometheus.pid"
  echo "prometheus started (pid $!)  -> http://localhost:9090"
fi

if running "$STATE/grafana.pid"; then
  echo "grafana already running (pid $(cat "$STATE/grafana.pid"))"
else
  GF_PATHS_DATA="$STATE/grafana-data" \
  GF_PATHS_LOGS="$STATE/logs" \
  GF_PATHS_PROVISIONING="$ROOT/grafana/provisioning" \
  GF_SERVER_HTTP_ADDR=127.0.0.1 \
  GF_AUTH_ANONYMOUS_ENABLED=true \
  GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer \
  PROMETHEUS_URL=http://localhost:9090 \
  AGENTLOAD_DASHBOARDS="$ROOT/dashboards" \
  grafana server --homepath "$(brew --prefix grafana)/share/grafana" \
    >"$STATE/logs/grafana.out" 2>&1 &
  echo $! >"$STATE/grafana.pid"
  echo "grafana started (pid $!)     -> http://localhost:3000"
fi
