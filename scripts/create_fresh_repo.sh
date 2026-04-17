#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/create_fresh_repo.sh /absolute/path/to/new-repo
#
# This script creates a brand-new git repository from the current project
# contents (without carrying over the old .git history).

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/new-repo"
  exit 1
fi

TARGET_DIR="$1"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${TARGET_DIR}" != /* ]]; then
  echo "Error: TARGET_DIR must be an absolute path."
  exit 1
fi

if [[ -e "${TARGET_DIR}" ]]; then
  echo "Error: target already exists: ${TARGET_DIR}"
  exit 1
fi

mkdir -p "${TARGET_DIR}"

# Copy project files excluding git metadata, caches, and local artifacts.
rsync -a \
  --exclude ".git" \
  --exclude "__pycache__" \
  --exclude ".pytest_cache" \
  --exclude ".mypy_cache" \
  --exclude ".ruff_cache" \
  --exclude "chatterbox_output" \
  --exclude "pretrained_models" \
  --exclude "*.pt" \
  --exclude "*.safetensors" \
  --exclude "*.wav" \
  "${SOURCE_DIR}/" "${TARGET_DIR}/"

cd "${TARGET_DIR}"
git init -b main
git add .
git commit -m "Initial commit: Vietnamese-only Standard + LJSpeech fork"

echo "✅ New repository created at: ${TARGET_DIR}"
echo "✅ Initial commit completed."
