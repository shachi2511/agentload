#!/usr/bin/env bash
# Stop the native Prometheus + Grafana started by scripts/obs_up.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
for name in prometheus grafana; do
  pidfile=".observability/$name.pid"
  if [ -f "$pidfile" ] && kill "$(cat "$pidfile")" 2>/dev/null; then
    echo "$name stopped"
  else
    echo "$name was not running"
  fi
  rm -f "$pidfile"
done
