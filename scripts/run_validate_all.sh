#!/usr/bin/env bash
# Full stability validation: 9 train scales x 3 shapes + 2 test scales x 3 shapes.
# Each group runs in its own process (Isaac Kit is single-instance per process).
#
# Usage:
#   bash DexReMoERebuild/scripts/run_validate_all.sh [num_envs]

set -u

NUM_ENVS="${1:-256}"
ROOT="/home/chen/my_project/DexReMoERebuild"
SRC_DIR="$ROOT/assets/grasp_cache_isaac"
OUT_DIR="$ROOT/assets/grasp_cache_isaac_validated"
LOG_DIR="$ROOT/logs/validation"
SCRIPT="$ROOT/scripts/validate_grasp_stability.py"

TRAIN_SCALES=(s07 s072 s074 s076 s078 s08 s082 s084 s086)
TEST_SCALES=(s066 s09)

mkdir -p "$OUT_DIR/train" "$OUT_DIR/test" "$LOG_DIR"
SUMMARY_FILE="$LOG_DIR/summaries.jsonl"
: > "$SUMMARY_FILE"

export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate isaaclab

run_group() {
    local split="$1"
    local scale_name="$2"
    local shape_id="$3"
    local npz="$SRC_DIR/$split/grasp_${scale_name}.npz"
    if [ ! -f "$npz" ]; then
        echo "[skip] missing $npz"
        return 0
    fi
    local log_file="/tmp/val_${split}_${scale_name}_${shape_id}.log"
    echo "===== group: $split / $scale_name / shape=$shape_id ====="
    python "$SCRIPT" \
        --npz "$npz" \
        --shape_id "$shape_id" \
        --split "$split" \
        --scale_name "$scale_name" \
        --output_dir "$OUT_DIR" \
        --log_dir "$LOG_DIR" \
        --num_envs "$NUM_ENVS" \
        > "$log_file" 2>&1
    local rc=$?
    # Extract summary from the finished log (no pipe buffering involved)
    grep -a 'SUMMARY_JSON' "$log_file" >> "$SUMMARY_FILE" || true
    return $rc
}

START=$(date +%s)
for scale in "${TRAIN_SCALES[@]}"; do
    for shape in 0 1 2; do
        run_group train "$scale" "$shape"
    done
done
for scale in "${TEST_SCALES[@]}"; do
    for shape in 0 1 2; do
        run_group test "$scale" "$shape"
    done
done
END=$(date +%s)
echo "ALL GROUPS DONE in $((END-START))s"

# Summarize into validation_meta.json
python - <<'PY'
import json, os
root = "/home/chen/my_project/DexReMoERebuild"
summ_file = os.path.join(root, "logs/validation/summaries.jsonl")
meta_path = os.path.join(root, "assets/grasp_cache_isaac_validated/validation_meta.json")

by_ss = {}
by_bucket = {
    "small":  {"scales": ["s07", "s072", "s074"], "n_total": 0, "n_pass": 0},
    "medium": {"scales": ["s076", "s078", "s08"], "n_total": 0, "n_pass": 0},
    "large":  {"scales": ["s082", "s084", "s086"], "n_total": 0, "n_pass": 0},
    "test":   {"scales": ["s066", "s09"], "n_total": 0, "n_pass": 0},
}

with open(summ_file) as f:
    for line in f:
        line = line.strip()
        if not line.startswith("SUMMARY_JSON "):
            continue
        s = json.loads(line[len("SUMMARY_JSON "):])
        key = s["group"]
        by_ss[key] = {k: s[k] for k in ["n_total", "n_pass", "pass_rate",
                                         "object_size_cm", "pos_tol_cm",
                                         "z_drop_tol_cm", "z_final_tol_cm"]}
        split = key.split("/")[0]
        scale = key.split("/")[1]
        if split == "test":
            bucket = "test"
        elif scale in by_bucket["small"]["scales"]:
            bucket = "small"
        elif scale in by_bucket["medium"]["scales"]:
            bucket = "medium"
        else:
            bucket = "large"
        by_bucket[bucket]["n_total"] += s["n_total"]
        by_bucket[bucket]["n_pass"] += s["n_pass"]

for b in by_bucket.values():
    b["pass_rate"] = (b["n_pass"] / b["n_total"]) if b["n_total"] else 0.0

meta = {
    "pd": {"stiffness": 3.0, "damping": 1.0},
    "steps": 60,
    "threshold_formula": {
        "pos_tol": "0.25 * size",
        "ori_tol_deg": 30.0,
        "z_drop_tol": "0.50 * size",
        "z_final_tol": "0.40 * size",
    },
    "by_shape_scale": by_ss,
    "by_size_bucket": by_bucket,
}
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"[meta] wrote {meta_path}")
PY
