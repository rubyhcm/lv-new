#!/usr/bin/env bash
# Generate split + preprocessed data for ONE dataset, exactly once, even when
# many jobs that share the same data directory start in parallel. A per-dataset
# lock directory ensures the first job does split+preprocess while the others
# wait, then reuse the freshly-generated files (no race, no stale skip). This
# uses mkdir instead of flock because macOS does not ship flock by default.
#
# Usage: bash scripts/ensure_data.sh configs/bgl.yaml
set -euo pipefail

CONFIG="${1:?usage: ensure_data.sh <config.yaml>}"

if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
else
    PYTHON_BIN=python3
fi

yq() { "$PYTHON_BIN" -c "import yaml; c=yaml.safe_load(open('$CONFIG'))
keys='$1'.split('.'); v=c
for k in keys: v=v[k]
print(v)"; }

TRAIN_RAW=$(yq paths.train_raw)
TEST_RAW=$(yq paths.test_raw)
TRAIN_NORMAL=$(yq paths.train_normal)
TEST_LOG=$(yq paths.test_log)

DATA_DIR="$(dirname "$TRAIN_NORMAL")"
mkdir -p "$DATA_DIR"
LOCK="$DATA_DIR/.prep.lock"

# Serialize the prep across all jobs sharing this data dir.
if [ -f "$LOCK" ]; then
    # Compatibility cleanup for the regular file created by the old flock
    # implementation. A file cannot represent an active lock in this script.
    rm -f "$LOCK"
fi
echo "==> [data] waiting for prep lock: $LOCK"
while ! mkdir "$LOCK" 2>/dev/null; do
    sleep 1
done
cleanup_lock() {
    rmdir "$LOCK"
}
trap cleanup_lock EXIT
echo "==> [data] acquired lock"

if [ -f "$TRAIN_RAW" ] && [ -f "$TEST_RAW" ]; then
    echo "    SKIP split (already generated: $TRAIN_RAW, $TEST_RAW)"
else
    "$PYTHON_BIN" -m lanobert.split --config "$CONFIG"
fi

if [ -f "$TRAIN_NORMAL" ]; then
    echo "    SKIP preprocess train (already generated: $TRAIN_NORMAL)"
else
    "$PYTHON_BIN" -m lanobert.preprocess --config "$CONFIG" --split train
fi

if [ -f "$TEST_LOG" ]; then
    echo "    SKIP preprocess test (already generated: $TEST_LOG)"
else
    "$PYTHON_BIN" -m lanobert.preprocess --config "$CONFIG" --split test
fi

rmdir "$LOCK"
trap - EXIT
echo "==> [data] released lock"
