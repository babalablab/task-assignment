#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/prune_experiment_configs.sh [--apply] [--yes] [--dir PATH]

Description:
  Keep only the experiment config files used in the paper experiments and
  delete other YAML files under config/experiment.

Options:
  --apply       Actually delete files (default: dry-run)
  --yes         Skip confirmation prompt when used with --apply
  --dir PATH    Target directory (default: config/experiment)
  -h, --help    Show this help

Examples:
  scripts/prune_experiment_configs.sh
  scripts/prune_experiment_configs.sh --apply
  scripts/prune_experiment_configs.sh --apply --yes
EOF
}

TARGET_DIR="config/experiment"
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

# Configs used in the paper experiments discussed above.
KEEP_FILES=(
  "tweet_eval_confusion.yaml"
  "tweet_eval_confusion_cost_const.yaml"
  "tweet_eval_learning_to_defer_assignment.yaml"
  "tweet_eval_icrowd_assignment.yaml"
  "spiral_different_test_num_confusion.yaml"
  "spiral_different_test_num_confusion_cost.yaml"
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

ALL_FILES=()
while IFS= read -r line; do
  ALL_FILES+=("$line")
done < <(find "$TARGET_DIR" -maxdepth 1 -type f \( -name "*.yaml" -o -name "*.yml" \) | sort)

TO_DELETE=()
for path in "${ALL_FILES[@]}"; do
  base="$(basename "$path")"
  if ! contains_keep_file "$base"; then
    TO_DELETE+=("$path")
  fi
done

echo "Target directory: $TARGET_DIR"
echo "Keep count: ${#KEEP_FILES[@]}"
echo "Delete candidate count: ${#TO_DELETE[@]}"
echo

if [[ ${#TO_DELETE[@]} -eq 0 ]]; then
  echo "Nothing to delete."
  exit 0
fi

printf '%s\n' "${TO_DELETE[@]}"
echo

if [[ "$APPLY" -ne 1 ]]; then
  echo "[dry-run] No files were deleted."
  echo "Run with --apply to delete these files."
  exit 0
fi

if [[ "$ASSUME_YES" -ne 1 ]]; then
  read -r -p "Delete these files? [y/N]: " answer
  if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
    echo "Aborted."
    exit 0
  fi
fi

for path in "${TO_DELETE[@]}"; do
  rm -- "$path"
done

echo "Deleted ${#TO_DELETE[@]} file(s)."
