#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

python3 "$REPO_ROOT/tools/repo_admin.py" analysis literature "$@"
python3 "$REPO_ROOT/tools/repo_admin.py" research ingest-openalex --latest
python3 "$REPO_ROOT/tools/repo_admin.py" research overview
