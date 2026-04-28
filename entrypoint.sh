#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${DATASET_DIR:-${INPUT_DIR:-/workspace/dataset}}"
INPUT_DIR="${INPUT_DIR:-$DATASET_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATASET_DIR/dn-splatter}"
IMAGE_DIR="${IMAGE_DIR:-$INPUT_DIR/images}"
METHOD="${METHOD:-dn-splatter}"
DATA_PARSER="${DATA_PARSER:-coolermap}"
STEPS="${STEPS:-4000}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-dnsplat_run}"
DOWNSCALE_FACTOR="${DOWNSCALE_FACTOR:-2}"
DEPTH_ENCODER="${DEPTH_ENCODER:-vits}"

RUN_COLMAP="${RUN_COLMAP:-auto}"              # auto | always | never
COLMAP_MATCHER="${COLMAP_MATCHER:-sequential}" # sequential | exhaustive
COLMAP_OVERLAP="${COLMAP_OVERLAP:-10}"
COLMAP_SINGLE_CAMERA="${COLMAP_SINGLE_CAMERA:-1}"
COLMAP_CAMERA_MODEL="${COLMAP_CAMERA_MODEL:-OPENCV}"
COLMAP_WORK_DIR="${COLMAP_WORK_DIR:-$INPUT_DIR/colmap_work}"
COLMAP_USE_GPU="${COLMAP_USE_GPU:-1}"
COLMAP_GPU_INDEX="${COLMAP_GPU_INDEX:--1}"
COLMAP_MAX_FEATURES="${COLMAP_MAX_FEATURES:-8192}"
COLMAP_MAX_IMAGE_SIZE="${COLMAP_MAX_IMAGE_SIZE:-2200}"
COLMAP_NUM_THREADS="${COLMAP_NUM_THREADS:-2}"
COLMAP_MIN_NUM_MATCHES="${COLMAP_MIN_NUM_MATCHES:-15}"
COLMAP_INIT_MIN_TRI_ANGLE="${COLMAP_INIT_MIN_TRI_ANGLE:-1.5}"
COLMAP_SPARSE_DIR="$INPUT_DIR/colmap/sparse/0"
LEGACY_SPARSE_DIR="$INPUT_DIR/sparse/0"

AUTO_MONO_DEPTH="${AUTO_MONO_DEPTH:-1}"
AUTO_ALIGN_DEPTH="${AUTO_ALIGN_DEPTH:-1}"
AUTO_SFM_DEPTH="${AUTO_SFM_DEPTH:-1}"

TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-$OUTPUT_DIR/ns_outputs}"
EXPORT_OUTPUT_DIR="${EXPORT_OUTPUT_DIR:-$OUTPUT_DIR/export}"
RUN_EXPORT="${RUN_EXPORT:-1}"
EXPORT_ON_INTERRUPT="${EXPORT_ON_INTERRUPT:-0}"
EXPORT_TIMEOUT="${EXPORT_TIMEOUT:-0}"
EXPORT_GAUSSIAN_SPLAT="${EXPORT_GAUSSIAN_SPLAT:-1}"
EXPORT_TSDF="${EXPORT_TSDF:-1}"
EXPORT_GS_MESH="${EXPORT_GS_MESH:-0}"
DEPTH_MODE="${DEPTH_MODE:-mono}"
LOAD_NORMALS="${LOAD_NORMALS:-False}"
LOAD_PCD_NORMALS="${LOAD_PCD_NORMALS:-False}"
TRAIN_COLMAP_PATH="${TRAIN_COLMAP_PATH:-colmap/sparse/0}"
TRAIN_IMAGES_PATH="${TRAIN_IMAGES_PATH:-$(basename "$IMAGE_DIR")}"
LOAD_3D_POINTS="${LOAD_3D_POINTS:-True}"
USE_DEPTH_LOSS="${USE_DEPTH_LOSS:-True}"
DEPTH_LOSS_TYPE="${DEPTH_LOSS_TYPE:-PearsonDepth}"
DEPTH_LAMBDA="${DEPTH_LAMBDA:-0.2}"
PEARSON_LAMBDA="${PEARSON_LAMBDA:-0.2}"
USE_NORMAL_LOSS="${USE_NORMAL_LOSS:-False}"
NORMAL_SUPERVISION="${NORMAL_SUPERVISION:-depth}"

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

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
  echo "==> Matcher: $COLMAP_MATCHER, single_camera=$COLMAP_SINGLE_CAMERA, camera_model=$COLMAP_CAMERA_MODEL, gpu=$COLMAP_USE_GPU, gpu_index=$COLMAP_GPU_INDEX"

  rm -rf "$COLMAP_WORK_DIR"
  mkdir -p "$COLMAP_WORK_DIR/sparse"

  local db="$COLMAP_WORK_DIR/database.db"

  colmap feature_extractor \
    --database_path "$db" \
    --image_path "$IMAGE_DIR" \
    --ImageReader.single_camera "$COLMAP_SINGLE_CAMERA" \
    --ImageReader.camera_model "$COLMAP_CAMERA_MODEL" \
    --SiftExtraction.use_gpu "$COLMAP_USE_GPU" \
    --SiftExtraction.gpu_index "$COLMAP_GPU_INDEX" \
    --SiftExtraction.max_num_features "$COLMAP_MAX_FEATURES" \
    --SiftExtraction.max_image_size "$COLMAP_MAX_IMAGE_SIZE" \
    --SiftExtraction.num_threads "$COLMAP_NUM_THREADS"

  if [ "$COLMAP_MATCHER" = "exhaustive" ]; then
    colmap exhaustive_matcher \
      --database_path "$db" \
      --SiftMatching.use_gpu "$COLMAP_USE_GPU" \
      --SiftMatching.gpu_index "$COLMAP_GPU_INDEX" \
      --SiftMatching.num_threads "$COLMAP_NUM_THREADS"
  else
    colmap sequential_matcher \
      --database_path "$db" \
      --SequentialMatching.overlap "$COLMAP_OVERLAP" \
      --SiftMatching.use_gpu "$COLMAP_USE_GPU" \
      --SiftMatching.gpu_index "$COLMAP_GPU_INDEX" \
      --SiftMatching.num_threads "$COLMAP_NUM_THREADS"
  fi

  colmap mapper \
    --database_path "$db" \
    --image_path "$IMAGE_DIR" \
    --output_path "$COLMAP_WORK_DIR/sparse" \
    --Mapper.ba_refine_focal_length 1 \
    --Mapper.ba_refine_principal_point 0 \
    --Mapper.ba_refine_extra_params 1 \
    --Mapper.min_num_matches "$COLMAP_MIN_NUM_MATCHES" \
    --Mapper.init_min_tri_angle "$COLMAP_INIT_MIN_TRI_ANGLE"

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
  --output-dir "$TRAIN_OUTPUT_DIR"
  --max-num-iterations "$STEPS"
  --pipeline.model.use-depth-loss "$USE_DEPTH_LOSS"
  --pipeline.model.depth-loss-type "$DEPTH_LOSS_TYPE"
  --pipeline.model.depth-lambda "$DEPTH_LAMBDA"
  --pipeline.model.pearson-lambda "$PEARSON_LAMBDA"
  --pipeline.model.use-normal-loss "$USE_NORMAL_LOSS"
  --pipeline.model.normal-supervision "$NORMAL_SUPERVISION"
  "$DATA_PARSER"
  --data "$INPUT_DIR"
  --depth-mode "$DEPTH_MODE"
  --load-normals "$LOAD_NORMALS"
  --load-pcd-normals "$LOAD_PCD_NORMALS"
  --colmap-path "$TRAIN_COLMAP_PATH"
  --images-path "$TRAIN_IMAGES_PATH"
  --downscale-factor "$DOWNSCALE_FACTOR"
  --load-3D-points "$LOAD_3D_POINTS"
)

export_done=0

find_config() {
  find "$TRAIN_OUTPUT_DIR" -name config.yml -type f -printf '%T@ %p\n' \
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

run_export_cmd() {
  echo "==> $*"
  if [ "$EXPORT_TIMEOUT" != "0" ] && command -v timeout >/dev/null 2>&1; then
    timeout "$EXPORT_TIMEOUT" "$@" || {
      local status=$?
      echo "Export command skončil statusem $status: $*"
      return 0
    }
  else
    "$@" || {
      local status=$?
      echo "Export command skončil statusem $status: $*"
      return 0
    }
  fi
}

do_export() {
  if [ "$RUN_EXPORT" != "1" ]; then
    echo "==> Export vypnutý přes RUN_EXPORT=$RUN_EXPORT."
    return 0
  fi

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

  if [ "$EXPORT_GAUSSIAN_SPLAT" = "1" ]; then
    mkdir -p "$EXPORT_OUTPUT_DIR/gaussian_splat"
    run_export_cmd ns-export gaussian-splat \
      --load-config "$config" \
      --output-dir "$EXPORT_OUTPUT_DIR/gaussian_splat"
  fi

  if [ "$EXPORT_TSDF" = "1" ]; then
    mkdir -p "$EXPORT_OUTPUT_DIR/mesh/tsdf"
    run_export_cmd ns-export tsdf \
      --load-config "$config" \
      --output-dir "$EXPORT_OUTPUT_DIR/mesh/tsdf"
  fi

  if [ "$EXPORT_GS_MESH" = "1" ]; then
    if command -v gs-mesh >/dev/null 2>&1; then
      mkdir -p "$EXPORT_OUTPUT_DIR/mesh/o3dtsdf" "$EXPORT_OUTPUT_DIR/mesh/gs_mesh_tsdf"
      run_export_cmd gs-mesh o3dtsdf \
        --load-config "$config" \
        --output-dir "$EXPORT_OUTPUT_DIR/mesh/o3dtsdf"

      run_export_cmd gs-mesh tsdf \
        --load-config "$config" \
        --output-dir "$EXPORT_OUTPUT_DIR/mesh/gs_mesh_tsdf"
    else
      echo "gs-mesh není dostupný, EXPORT_GS_MESH přeskočen."
    fi
  fi

  echo "==> Export hotový: $EXPORT_OUTPUT_DIR"
}

on_signal() {
  trap - INT TERM
  echo ""
  echo "==> Zachycen Ctrl-C / stop. Ukončuji trénink..."
  if [ -n "${TRAIN_PID:-}" ]; then
    kill -INT "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" 2>/dev/null || true
  fi

  if [ "$EXPORT_ON_INTERRUPT" = "1" ]; then
    echo "==> EXPORT_ON_INTERRUPT=1, exportuji poslední checkpoint..."
    do_export
  else
    echo "==> Export na Ctrl-C přeskočen. Checkpointy zůstávají v: $TRAIN_OUTPUT_DIR"
  fi

  exit 0
}

trap on_signal INT TERM

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

do_export
