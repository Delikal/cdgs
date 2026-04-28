#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${INPUT_DIR:-/data/input}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/output}"
IMAGE_DIR="${IMAGE_DIR:-$INPUT_DIR/images}"
METHOD="${METHOD:-dn-splatter}"
DATA_PARSER="${DATA_PARSER:-coolermap}"
STEPS="${STEPS:-4000}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-dnsplat_run}"
DOWNSCALE_FACTOR="${DOWNSCALE_FACTOR:-4}"
DEPTH_ENCODER="${DEPTH_ENCODER:-vits}"

RUN_COLMAP="${RUN_COLMAP:-auto}"              # auto | always | never
COLMAP_MATCHER="${COLMAP_MATCHER:-sequential}" # sequential | exhaustive
COLMAP_OVERLAP="${COLMAP_OVERLAP:-10}"
COLMAP_SINGLE_CAMERA="${COLMAP_SINGLE_CAMERA:-1}"
COLMAP_CAMERA_MODEL="${COLMAP_CAMERA_MODEL:-OPENCV}"
COLMAP_WORK_DIR="${COLMAP_WORK_DIR:-$INPUT_DIR/colmap_work}"
COLMAP_SPARSE_DIR="$INPUT_DIR/colmap/sparse/0"
LEGACY_SPARSE_DIR="$INPUT_DIR/sparse/0"

AUTO_MONO_DEPTH="${AUTO_MONO_DEPTH:-1}"
AUTO_ALIGN_DEPTH="${AUTO_ALIGN_DEPTH:-1}"
AUTO_SFM_DEPTH="${AUTO_SFM_DEPTH:-1}"

mkdir -p "$OUTPUT_DIR"

source /opt/conda/etc/profile.d/conda.sh
conda activate nerfstudio
export QT_QPA_PLATFORM=offscreen
export DISPLAY=
export PYTHONUNBUFFERED=1
export DEPTH_ENCODER

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing command: $1" >&2; exit 1; }
}

points3d_count() {
  local f="$1"
  python - "$f" <<'PY'
import struct, sys
p = sys.argv[1]
try:
    with open(p, 'rb') as fh:
        raw = fh.read(8)
    print(struct.unpack('<Q', raw)[0] if len(raw) == 8 else 0)
except FileNotFoundError:
    print(0)
PY
}

has_valid_colmap() {
  local dir="$1"
  [ -f "$dir/cameras.bin" ] || return 1
  [ -f "$dir/images.bin" ] || return 1
  [ -f "$dir/points3D.bin" ] || return 1
  local n
  n="$(points3d_count "$dir/points3D.bin")"
  [ "${n:-0}" -gt 0 ]
}

sync_colmap_outputs() {
  mkdir -p "$COLMAP_SPARSE_DIR" "$LEGACY_SPARSE_DIR"
  rsync -a --delete "$1/" "$COLMAP_SPARSE_DIR/"
  rsync -a --delete "$1/" "$LEGACY_SPARSE_DIR/"
  echo "==> COLMAP model prepared: $COLMAP_SPARSE_DIR"
  echo "==> points3D count: $(points3d_count "$COLMAP_SPARSE_DIR/points3D.bin")"
}

find_best_colmap_model() {
  local sparse_root="$1"
  python - "$sparse_root" <<'PY'
from pathlib import Path
import struct, sys
root = Path(sys.argv[1])
best = None
best_n = -1
for d in sorted(root.iterdir()) if root.exists() else []:
    f = d / 'points3D.bin'
    if not f.exists():
        continue
    try:
        n = struct.unpack('<Q', f.read_bytes()[:8])[0]
    except Exception:
        n = 0
    if n > best_n:
        best = d
        best_n = n
if best is None or best_n <= 0:
    sys.exit(1)
print(best)
PY
}

run_colmap_pipeline() {
  need_cmd colmap

  if [ ! -d "$IMAGE_DIR" ]; then
    echo "Images folder does not exist: $IMAGE_DIR" >&2
    exit 1
  fi

  echo "==> Spouštím COLMAP pipeline z: $IMAGE_DIR"
  echo "==> Matcher: $COLMAP_MATCHER, single_camera=$COLMAP_SINGLE_CAMERA, camera_model=$COLMAP_CAMERA_MODEL"

  rm -rf "$COLMAP_WORK_DIR"
  mkdir -p "$COLMAP_WORK_DIR/sparse"

  local db="$COLMAP_WORK_DIR/database.db"

  colmap feature_extractor \
    --database_path "$db" \
    --image_path "$IMAGE_DIR" \
    --ImageReader.single_camera "$COLMAP_SINGLE_CAMERA" \
    --ImageReader.camera_model "$COLMAP_CAMERA_MODEL" \
    --SiftExtraction.use_gpu "$COLMAP_USE_GPU" \
    --SiftExtraction.max_num_features "${COLMAP_MAX_FEATURES:-8192}" \
    --SiftExtraction.max_image_size "${COLMAP_MAX_IMAGE_SIZE:-2200}" \
    --SiftExtraction.num_threads "${COLMAP_NUM_THREADS:-2}"

  if [ "$COLMAP_MATCHER" = "exhaustive" ]; then
    colmap exhaustive_matcher \
      --database_path "$db" \
      --SiftMatching.use_gpu "$COLMAP_USE_GPU" \
      --SiftMatching.num_threads "${COLMAP_NUM_THREADS:-2}"
  else
    colmap sequential_matcher \
      --database_path "$db" \
      --SequentialMatching.overlap "$COLMAP_OVERLAP" \
      --SiftMatching.use_gpu "$COLMAP_USE_GPU" \
      --SiftMatching.num_threads "${COLMAP_NUM_THREADS:-2}"
  fi

  colmap mapper \
    --database_path "$db" \
    --image_path "$IMAGE_DIR" \
    --output_path "$COLMAP_WORK_DIR/sparse" \
    --Mapper.ba_refine_focal_length 1 \
    --Mapper.ba_refine_principal_point 0 \
    --Mapper.ba_refine_extra_params 1 \
    --Mapper.min_num_matches "${COLMAP_MIN_NUM_MATCHES:-15}" \
    --Mapper.init_min_tri_angle "${COLMAP_INIT_MIN_TRI_ANGLE:-1.5}"

  local best
  if ! best="$(find_best_colmap_model "$COLMAP_WORK_DIR/sparse")"; then
    echo "COLMAP nevytvořil validní sparse model s points3D.bin." >&2
    echo "Zkus: -e COLMAP_MATCHER=exhaustive nebo zvýšit overlap: -e COLMAP_OVERLAP=20" >&2
    exit 1
  fi

  sync_colmap_outputs "$best"
}

prepare_colmap() {
  if [ "$RUN_COLMAP" = "always" ]; then
    run_colmap_pipeline
    return
  fi

  if [ "$RUN_COLMAP" = "never" ]; then
    if has_valid_colmap "$COLMAP_SPARSE_DIR"; then
      return
    fi
    if has_valid_colmap "$LEGACY_SPARSE_DIR"; then
      sync_colmap_outputs "$LEGACY_SPARSE_DIR"
      return
    fi
    echo "RUN_COLMAP=never, ale validní COLMAP model neexistuje." >&2
    exit 1
  fi

  if has_valid_colmap "$COLMAP_SPARSE_DIR"; then
    echo "==> Používám existující validní COLMAP: $COLMAP_SPARSE_DIR"
    return
  fi

  if has_valid_colmap "$LEGACY_SPARSE_DIR"; then
    echo "==> Přesouvám existující validní sparse/0 do colmap/sparse/0"
    sync_colmap_outputs "$LEGACY_SPARSE_DIR"
    return
  fi

  echo "==> Validní COLMAP pointcloud nenalezen, spouštím rekonstrukci."
  run_colmap_pipeline
}

generate_mono_depth() {
  if [ "$AUTO_MONO_DEPTH" != "1" ]; then
    return
  fi
  if find "$INPUT_DIR/mono_depth" -name "*.npy" -type f 2>/dev/null | grep -q .; then
    echo "==> mono_depth už existuje, přeskakuji DepthAnythingV2."
    return
  fi
  echo "==> Generuji mono_depth přes DepthAnythingV2 ($DEPTH_ENCODER)..."
  INPUT_IMAGES="$IMAGE_DIR" \
  OUTPUT_DEPTH="$INPUT_DIR/mono_depth" \
  python /generate_depth.py
}

generate_sfm_and_aligned_depth() {
  if [ "$AUTO_SFM_DEPTH" != "1" ] && [ "$AUTO_ALIGN_DEPTH" != "1" ]; then
    return
  fi

  local args=(
    python /opt/dn-splatter/dn_splatter/scripts/align_depth.py
    --data "$INPUT_DIR"
    --sparse-path colmap/sparse/0
    --img-dir-name "$(basename "$IMAGE_DIR")"
    --skip-mono-depth-creation
    --align-method closed_form
  )

  if [ "$AUTO_SFM_DEPTH" != "1" ]; then
    args+=(--skip-colmap-to-depths)
  fi
  if [ "$AUTO_ALIGN_DEPTH" != "1" ]; then
    args+=(--skip-alignment)
  fi

  if [ "$AUTO_ALIGN_DEPTH" = "1" ] && find "$INPUT_DIR/mono_depth" -name "*_aligned.npy" -type f 2>/dev/null | grep -q .; then
    echo "==> Aligned mono_depth už existuje, přeskakuji align_depth."
    return
  fi

  echo "==> Generuji sfm_depths a aligned mono_depth přes COLMAP pointcloud..."
  "${args[@]}"
}

prepare_colmap
if [ "$METHOD" = "dn-splatter" ]; then
  generate_mono_depth
  generate_sfm_and_aligned_depth
fi

TRAIN_CMD=(
  ns-train "$METHOD"
  --experiment-name "$EXPERIMENT_NAME"
  --output-dir "$OUTPUT_DIR/ns_outputs"
  --max-num-iterations "$STEPS"
  --pipeline.model.use-depth-loss True
  --pipeline.model.depth-loss-type PearsonDepth
  --pipeline.model.depth-lambda "${DEPTH_LAMBDA:-0.2}"
  --pipeline.model.pearson-lambda "${PEARSON_LAMBDA:-0.2}"
  --pipeline.model.use-normal-loss False
  --pipeline.model.normal-supervision depth
  "$DATA_PARSER"
  --data "$INPUT_DIR"
  --depth-mode mono
  --load-normals False
  --load-pcd-normals False
  --colmap-path colmap/sparse/0
  --images-path images
  --downscale-factor 2
  --load-3D-points True
)

export_done=0

find_config() {
  find "$OUTPUT_DIR/ns_outputs" -name config.yml -type f -printf '%T@ %p\n' \
    | sort -nr \
    | head -n1 \
    | cut -d' ' -f2-
}

has_checkpoint() {
  local config="$1"
  local checkpoint_dir
  checkpoint_dir="$(dirname "$config")/nerfstudio_models"
  find "$checkpoint_dir" \( -name "*.ckpt" -o -name "*.pth" \) -type f 2>/dev/null | grep -q .
}

do_export() {
  if [ "$export_done" = "1" ]; then
    return 0
  fi
  export_done=1

  echo ""
  echo "==> Exportuji model..."

  local config
  config="$(find_config || true)"

  if [ -z "${config:-}" ]; then
    echo "Nenalezen config.yml, export není možný."
    return 0
  fi

  if ! has_checkpoint "$config"; then
    echo "Checkpoint neexistuje, export přeskočen: $config"
    return 0
  fi

  echo "Používám config: $config"

  mkdir -p "$OUTPUT_DIR/export/gaussian_splat" "$OUTPUT_DIR/export/mesh"

  ns-export gaussian-splat \
    --load-config "$config" \
    --output-dir "$OUTPUT_DIR/export/gaussian_splat" || true

  ns-export tsdf \
    --load-config "$config" \
    --output-dir "$OUTPUT_DIR/export/mesh/tsdf" || true

  if command -v gs-mesh >/dev/null 2>&1; then
    gs-mesh o3dtsdf \
      --load-config "$config" \
      --output-dir "$OUTPUT_DIR/export/mesh/o3dtsdf" || true

    gs-mesh tsdf \
      --load-config "$config" \
      --output-dir "$OUTPUT_DIR/export/mesh/gs_mesh_tsdf" || true
  fi

  echo "==> Export hotový: $OUTPUT_DIR/export"
}

on_signal() {
  trap - INT TERM EXIT
  echo ""
  echo "==> Zachycen Ctrl-C / stop. Ukončuji trénink a exportuji poslední checkpoint..."
  if [ -n "${TRAIN_PID:-}" ]; then
    kill -INT "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" 2>/dev/null || true
  fi
  do_export
  exit 0
}

trap on_signal INT TERM
trap do_export EXIT

echo "==> Spouštím:"
printf ' %q' "${TRAIN_CMD[@]}"
echo ""

# yes y kvůli případnému Nerfstudio downscale promptu
set +e
yes y | "${TRAIN_CMD[@]}" &
TRAIN_PID=$!
wait "$TRAIN_PID"
TRAIN_STATUS=$?
set -e

if [ "$TRAIN_STATUS" -ne 0 ]; then
  echo "Trénink skončil chybou: $TRAIN_STATUS"
  exit "$TRAIN_STATUS"
fi
