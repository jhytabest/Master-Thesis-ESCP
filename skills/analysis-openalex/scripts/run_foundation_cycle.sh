#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

python3 "$REPO_ROOT/tools/repo_admin.py" analysis literature "$@"
python3 "$REPO_ROOT/tools/repo_admin.py" research ingest-openalex --latest --min-citations 50
python3 "$REPO_ROOT/tools/repo_admin.py" research snapshot
python3 "$REPO_ROOT/tools/repo_admin.py" research overview

echo "Foundation cycle complete: literature mapped, ingested with quality gates, snapshot generated, overview refreshed."
