#!/usr/bin/env bash
# Sync LPFchan/Chat's built web SPA (dist/) into ./state/chat-web/ in place.
# Uses rsync with trailing slashes so ./state/chat-web's own directory inode
# is never unlinked or replaced while bind-mounted into the grimoire container.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHAT_REPO="${CHAT_REPO:-$REPO_ROOT/../Chat}"
DEST="$REPO_ROOT/state/chat-web"

BUILD=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [--build]

Sync LPFchan/Chat's built web assets into state/chat-web/ in place.

Options:
  --build      Run 'npm run build' in \$CHAT_REPO before syncing
  -h, --help   Show this help message

Environment:
  CHAT_REPO    Path to LPFchan/Chat checkout (default: \$REPO_ROOT/../Chat)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)
      BUILD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v rsync >/dev/null 2>&1; then
  echo "error: rsync is required but not found in PATH" >&2
  exit 1
fi

if [[ ! -d "$CHAT_REPO" ]]; then
  echo "error: CHAT_REPO directory not found: $CHAT_REPO" >&2
  exit 1
fi

if [[ "$BUILD" -eq 1 ]]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "error: npm is required for --build but not found in PATH" >&2
    exit 1
  fi
  echo "Building Chat web SPA in $CHAT_REPO..."
  (cd "$CHAT_REPO" && npm run build)
fi

SRC="$CHAT_REPO/dist"
if [[ ! -f "$SRC/index.html" ]]; then
  echo "error: built Chat web SPA not found at $SRC/index.html" >&2
  echo "hint: run with --build or run 'npm run build' in $CHAT_REPO first." >&2
  exit 1
fi

mkdir -p "$DEST"
if [[ ! -w "$DEST" ]]; then
  echo "error: destination directory is not writable: $DEST" >&2
  echo "hint: if Docker created $DEST as root, fix ownership before syncing." >&2
  exit 1
fi

# Trailing slashes keep $DEST's own directory inode intact for live bind mounts;
# --delete-after transfers new assets and index.html before pruning old hashed files.
rsync -a --delete --delete-after "$SRC/" "$DEST/"

FILE_COUNT="$(find "$DEST" -type f | wc -l | tr -d ' ')"
INDEX_BYTES="$(wc -c < "$DEST/index.html" | tr -d ' ')"
echo "Synced $FILE_COUNT files to $DEST (index.html: ${INDEX_BYTES} bytes)"
