#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/commentout_unused_dataset_configs.sh [--apply] [--yes] [--dir PATH]

Description:
  Keep only dataset configs required by the selected paper experiments and
  comment out all other YAML files under config/dataset.

Options:
  --apply       Actually comment out files (default: dry-run)
  --yes         Skip confirmation prompt when used with --apply
  --dir PATH    Target directory (default: config/dataset)
  -h, --help    Show this help

Examples:
  scripts/commentout_unused_dataset_configs.sh
  scripts/commentout_unused_dataset_configs.sh --apply
  scripts/commentout_unused_dataset_configs.sh --apply --yes
EOF
}

TARGET_DIR="config/dataset"
APPLY=0
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      APPLY=1
      shift
      ;;
    --yes)
      ASSUME_YES=1
      shift
      ;;
    --dir)
      TARGET_DIR="${2:-}"
      if [[ -z "$TARGET_DIR" ]]; then
        echo "Error: --dir requires a path" >&2
        exit 1
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "Error: directory not found: $TARGET_DIR" >&2
  exit 1
fi

# Dataset configs used by the previously selected experiments.
KEEP_FILES=(
  "tweet_eval.yaml"
  "tweet_eval_vocab.yaml"
  "spiral_test_num.yaml"
)

contains_keep_file() {
  local needle="$1"
  local item
  for item in "${KEEP_FILES[@]}"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

is_fully_commented_or_empty() {
  local path="$1"
  awk '
    /^[[:space:]]*$/ { next }
    /^[[:space:]]*#/ { next }
    { exit 1 }
    END { exit 0 }
  ' "$path"
}

comment_out_file_inplace() {
  local path="$1"
  local tmp
  tmp="$(mktemp)"
  sed 's/^/# /' "$path" > "$tmp"
  mv "$tmp" "$path"
}

ALL_FILES=()
while IFS= read -r line; do
  ALL_FILES+=("$line")
done < <(find "$TARGET_DIR" -maxdepth 1 -type f \( -name "*.yaml" -o -name "*.yml" \) | sort)

TO_COMMENT=()
for path in "${ALL_FILES[@]}"; do
  base="$(basename "$path")"
  if ! contains_keep_file "$base"; then
    TO_COMMENT+=("$path")
  fi
done

echo "Target directory: $TARGET_DIR"
echo "Keep count: ${#KEEP_FILES[@]}"
echo "Comment-out candidate count: ${#TO_COMMENT[@]}"
echo

if [[ ${#TO_COMMENT[@]} -eq 0 ]]; then
  echo "Nothing to comment out."
  exit 0
fi

printf '%s\n' "${TO_COMMENT[@]}"
echo

if [[ "$APPLY" -ne 1 ]]; then
  echo "[dry-run] No files were modified."
  echo "Run with --apply to comment out these files."
  exit 0
fi

if [[ "$ASSUME_YES" -ne 1 ]]; then
  read -r -p "Comment out these files? [y/N]: " answer
  if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
    echo "Aborted."
    exit 0
  fi
fi

modified=0
skipped=0
for path in "${TO_COMMENT[@]}"; do
  if is_fully_commented_or_empty "$path"; then
    skipped=$((skipped + 1))
    continue
  fi
  comment_out_file_inplace "$path"
  modified=$((modified + 1))
done

echo "Modified: $modified file(s)"
echo "Skipped (already commented/empty): $skipped file(s)"
