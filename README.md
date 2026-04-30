# CDGS / dn-splatter pipeline

Docker image spouští kompletní přípravu datasetu a trénink:

- COLMAP sparse rekonstrukci s CUDA podporou,
- mono depth přes Depth Anything V2,
- zarovnání depth map přes `dn-splatter`,
- `ns-train` a následný export gaussian splatu / meshe.

## Build image

```bash
docker build -t cdgs .
```

COLMAP se v image builduje ze zdrojáků s CUDA podporou. Běžný Ubuntu balíček `colmap` CUDA podporu typicky nemá, proto se nepoužívá apt balíček. Viz oficiální COLMAP instalace: https://colmap.github.io/install.html

Defaultně se builduje `COLMAP_VERSION=3.13.0`, protože rig workflow používá command `rig_configurator`, který ve starém COLMAP `3.9.1` není.

Výchozí CUDA architektura je `89`. Pro jinou kartu ji změň při buildu:

```bash
docker build \
  --build-arg COLMAP_CUDA_ARCHITECTURES=86 \
  -t cdgs .
```

## Default dataset layout

Když nic nenastavíš, používá se `/workspace/dataset`. Při běžném spuštění si ho namapuj na lokální složku `dataset` v rootu projektu:

```bash
mkdir -p dataset/images
```

Vstupní fotky dej do:

```text
dataset/images
```

Pipeline potom vytvoří například:

```text
dataset/
  images/
  colmap/sparse/0/
  sparse/0/
  mono_depth/
  sfm_depths/
  dn-splatter/
    ns_outputs/
    export/
```

## Run

```bash
docker run --rm --gpus all \
  -v "$PWD/dataset:/workspace/dataset" \
  cdgs
```

Výstup tréninku a exportu bude defaultně v `dataset/dn-splatter`.

## Custom output dir

Dataset může zůstat v `dataset`, ale výstup tréninku můžeš poslat jinam:

```bash
mkdir -p output

docker run --rm --gpus all \
  -v "$PWD/dataset:/workspace/dataset" \
  -v "$PWD/output:/workspace/output" \
  -e OUTPUT_DIR=/workspace/output \
  cdgs
```

Pak vznikne:

```text
output/
  ns_outputs/
  export/
```

## Důležité proměnné

| Proměnná | Default | Popis |
| --- | --- | --- |
| `DATASET_DIR` | `/workspace/dataset` | Root datasetu. Obsahuje `images`, `colmap`, `mono_depth`, atd. |
| `INPUT_DIR` | `$DATASET_DIR` | Data root předaný do parseru. Zachováno kvůli kompatibilitě. |
| `IMAGE_DIR` | `$INPUT_DIR/images` | Složka se vstupními obrázky. |
| `OUTPUT_DIR` | `$DATASET_DIR/dn-splatter` | Root výstupů tréninku a exportu. |
| `TRAIN_OUTPUT_DIR` | `$OUTPUT_DIR/ns_outputs` | Nerfstudio output dir. |
| `EXPORT_OUTPUT_DIR` | `$OUTPUT_DIR/export` | Export gaussian splatu / meshe. |
| `RUN_EXPORT` | `1` | Spustí export po úspěšném doběhnutí tréninku. |
| `EXPORT_ON_INTERRUPT` | `0` | Když je `1`, spustí export i po Ctrl-C. Defaultně se Ctrl-C nezasekne v exportu. |
| `EXPORT_TIMEOUT` | `0` | Timeout pro jeden export příkaz, např. `30m`. `0` timeout vypíná. |
| `EXPORT_GAUSSIAN_SPLAT` | `1` | Export barevného gaussian splatu. |
| `EXPORT_TSDF` | `1` | Export TSDF meshe přes `ns-export`. |
| `EXPORT_GS_MESH` | `0` | Volitelné `gs-mesh` exporty, bývají pomalejší. |
| `RUN_COLMAP` | `auto` | `auto`, `always`, nebo `never`. |
| `COLMAP_USE_GPU` | `1` | Zapne GPU pro SIFT extraction a matching. |
| `COLMAP_GPU_INDEX` | `-1` | GPU index pro COLMAP. `-1` nechá COLMAP vybrat default. |
| `COLMAP_MATCHER` | `sequential` | `sequential` nebo `exhaustive`. |
| `COLMAP_OVERLAP` | `10` | Overlap pro sequential matcher. |
| `COLMAP_RIG_CONFIG` | auto | Cesta k COLMAP rig configu. Když není nastavena a existuje `dataset/rig_config.json`, použije se automaticky. |
| `COLMAP_SINGLE_CAMERA_PER_FOLDER` | `auto` | V rig režimu se automaticky nastaví na `1`, jinak se použije `COLMAP_SINGLE_CAMERA`. |
| `COLMAP_BA_REFINE_SENSOR_FROM_RIG` | `0` | Když se používá rig, drží relativní pózy kamer fixní. |
| `COLMAP_BA_REFINE_FOCAL_LENGTH` | `1` | Předáno do mapperu jako `Mapper.ba_refine_focal_length`. |
| `COLMAP_BA_REFINE_PRINCIPAL_POINT` | `0` | Předáno do mapperu jako `Mapper.ba_refine_principal_point`. |
| `COLMAP_BA_REFINE_EXTRA_PARAMS` | `1` | Předáno do mapperu jako `Mapper.ba_refine_extra_params`. |
| `STEPS` | `4000` | Počet trénovacích iterací. |
| `DOWNSCALE_FACTOR` | `2` | Hodnota předaná do `ns-train --downscale-factor`. |
| `DEPTH_ENCODER` | `vits` | Depth Anything encoder. Image obsahuje checkpointy `vits` a `vitb`. |

## COLMAP rig

Pro pevný sloup kamer dej `rig_config.json` přímo do rootu datasetu:

```text
dataset/
  images/
    2026042001/
      119/
        ls00pri/
        ls01pri/
        rs00pri/
  rig_config.json
```

Pipeline ho najde automaticky a před matchingem zavolá:

```bash
colmap rig_configurator \
  --database_path "$COLMAP_WORK_DIR/database.db" \
  --rig_config_path "$INPUT_DIR/rig_config.json"
```

V rig režimu se zároveň použije `ImageReader.single_camera_per_folder=1`, takže každá složka kamery má vlastní intrinsics. Relativní pozice kamer v rigu se defaultně neoptimalizují:

```text
COLMAP_BA_REFINE_SENSOR_FROM_RIG=0
```

Příklad spuštění s rigem a exhaustive matchingem:

```bash
docker run --rm --gpus all \
  -v "$PWD/dataset:/workspace/dataset" \
  -e RUN_COLMAP=always \
  -e COLMAP_MATCHER=exhaustive \
  cdgs
```

Pokud je config jinde než `dataset/rig_config.json`, předej cestu explicitně:

```bash
docker run --rm --gpus all \
  -v "$PWD/dataset:/workspace/dataset" \
  -e COLMAP_RIG_CONFIG=/workspace/dataset/configs/rig_config.json \
  cdgs
```

### Funkční rig trénink se SfM depth alignmentem

Tahle varianta se osvědčila pro rig dataset: COLMAP běží s rigem, používá konzervativnější `SIMPLE_RADIAL` model, exhaustive matching a potom se z COLMAP sparse point cloudu generují `sfm_depths`, podle kterých se zarovnávají mono depth mapy.

```bash
docker run --rm --gpus all \
  -p 7007:7007 \
  -v "$PWD/dataset:/workspace/dataset" \
  -e RUN_COLMAP=always \
  -e COLMAP_USE_RIG=1 \
  -e COLMAP_MATCHER=exhaustive \
  -e COLMAP_CAMERA_MODEL=SIMPLE_RADIAL \
  -e COLMAP_RIG_CONFIG=/workspace/dataset/rig_config.json \
  -e COLMAP_BA_REFINE_SENSOR_FROM_RIG=0 \
  -e COLMAP_MIN_NUM_MATCHES=8 \
  -e COLMAP_INIT_MIN_TRI_ANGLE=0.5 \
  -e ALIGN_DEPTH_BATCH_SIZE=1 \
  -e RUN_EXPORT=0 \
  cdgs
```

Když už máš dobrý COLMAP model v `dataset/colmap/sparse/0` a chceš jen znovu dopočítat mono/SfM depth alignment a trénink, použij:

```bash
rm -rf dataset/sfm_depths
find dataset/mono_depth -name "*_aligned.npy" -delete 2>/dev/null || true

docker run --rm --gpus all \
  -p 7007:7007 \
  -v "$PWD/dataset:/workspace/dataset" \
  -e RUN_COLMAP=never \
  -e ALIGN_DEPTH_BATCH_SIZE=1 \
  -e RUN_EXPORT=0 \
  cdgs
```

Pro rig dataset nenechávej v `dataset/images` fotky navíc, pokud je nechceš rekonstruovat. `rig_config.json` dataset nefiltruje; pouze říká COLMAPu, které prefixy patří do rigu.

## Nastavení `ns-train`

Hodnoty, které byly dřív natvrdo v `TRAIN_CMD`, jde přepsat env proměnnými:

| Proměnná | Default |
| --- | --- |
| `METHOD` | `dn-splatter` |
| `DATA_PARSER` | `coolermap` |
| `EXPERIMENT_NAME` | `dnsplat_run` |
| `DEPTH_MODE` | `mono` |
| `LOAD_NORMALS` | `False` |
| `LOAD_PCD_NORMALS` | `False` |
| `TRAIN_COLMAP_PATH` | `colmap/sparse/0` |
| `TRAIN_IMAGES_PATH` | basename z `IMAGE_DIR`, defaultně `images` |
| `LOAD_3D_POINTS` | `True` |
| `USE_DEPTH_LOSS` | `True` |
| `DEPTH_LOSS_TYPE` | `PearsonDepth` |
| `DEPTH_LAMBDA` | `0.2` |
| `PEARSON_LAMBDA` | `0.2` |
| `USE_NORMAL_LOSS` | `False` |
| `NORMAL_SUPERVISION` | `depth` |

Příklad:

```bash
docker run --rm --gpus all \
  -v "$PWD/dataset:/workspace/dataset" \
  -e STEPS=8000 \
  -e COLMAP_MATCHER=exhaustive \
  -e DOWNSCALE_FACTOR=4 \
  cdgs
```

## Exporty a Ctrl-C

Po úspěšném doběhnutí tréninku se defaultně exportuje:

```text
dataset/dn-splatter/export/gaussian_splat/
dataset/dn-splatter/export/mesh/tsdf/
```

Na Ctrl-C se export defaultně nepouští, aby se běh nezasekl v pomalém mesh exportu. Checkpointy zůstanou v `dataset/dn-splatter/ns_outputs`.

Pokud chceš jen gaussian splat a nechceš mesh:

```bash
docker run --rm --gpus all \
  -v "$PWD/dataset:/workspace/dataset" \
  -e EXPORT_TSDF=0 \
  -e EXPORT_GS_MESH=0 \
  cdgs
```

Pokud chceš exportovat i po Ctrl-C:

```bash
docker run --rm --gpus all \
  -v "$PWD/dataset:/workspace/dataset" \
  -e EXPORT_ON_INTERRUPT=1 \
  -e EXPORT_TIMEOUT=30m \
  cdgs
```

Gaussian splat export je `.ply` se splaty a barvami, ne klasický mesh s UV texturami. COLMAP `.ply` je jen sparse point cloud, takže nestačí jako finální vizuální výstup. Klasické texturované OBJ/mesh UV textury tahle pipeline defaultně negeneruje; mesh exporty jsou rekonstrukce z trénovaného modelu.

### Ruční export po tréninku

Najdi poslední trénovací config:

```bash
CONFIG="$(find dataset/dn-splatter/ns_outputs -name config.yml -type f -printf '%T@ %p\n' \
  | sort -nr \
  | head -n1 \
  | cut -d' ' -f2-)"

echo "$CONFIG"
```

Export Gaussian splatu:

```bash
docker run --rm --gpus all \
  -v "$PWD/dataset:/workspace/dataset" \
  --entrypoint bash \
  cdgs -lc "
    source /opt/conda/etc/profile.d/conda.sh
    conda activate nerfstudio
    export TORCH_DISABLE_DYNAMO=1
    export TORCHDYNAMO_DISABLE=1

    ns-export gaussian-splat \
      --load-config /workspace/${CONFIG} \
      --output-dir /workspace/dataset/dn-splatter/export/gaussian_splat
  "
```

Výstup je `.ply` se splaty, typicky vhodný pro SuperSplat / Gaussian Splat viewer:

```text
dataset/dn-splatter/export/gaussian_splat/
```

Rychlý mesh přes Open3D TSDF:

```bash
docker run --rm --gpus all \
  -v "$PWD/dataset:/workspace/dataset" \
  --entrypoint bash \
  cdgs -lc "
    source /opt/conda/etc/profile.d/conda.sh
    conda activate nerfstudio
    export TORCH_DISABLE_DYNAMO=1
    export TORCHDYNAMO_DISABLE=1

    gs-mesh o3dtsdf \
      --load-config /workspace/${CONFIG} \
      --output-dir /workspace/dataset/dn-splatter/export/mesh/o3dtsdf \
      --voxel-size 0.004 \
      --sdf-truc 0.02 \
      --depth-trunc 4
  "
```

Pozor: argument se v aktuální `gs-mesh` verzi jmenuje `--sdf-truc`, ne `--sdf-trunc`.

Detailnější Poisson mesh z gaussianů:

```bash
docker run --rm --gpus all \
  -v "$PWD/dataset:/workspace/dataset" \
  --entrypoint bash \
  cdgs -lc "
    source /opt/conda/etc/profile.d/conda.sh
    conda activate nerfstudio
    export TORCH_DISABLE_DYNAMO=1
    export TORCHDYNAMO_DISABLE=1
    export OMP_NUM_THREADS=12

    gs-mesh gaussians \
      --load-config /workspace/${CONFIG} \
      --output-dir /workspace/dataset/dn-splatter/export/mesh/gaussians_poisson_d13_detail \
      --min-opacity 0.01 \
      --down-sample-voxel 0.001 \
      --outlier-removal True \
      --std-ratio 3.0 \
      --poisson-depth 13
  "
```

Ještě agresivnější detail, ale s vyšším rizikem šumu a delší dobou běhu:

```bash
docker run --rm --gpus all \
  -v "$PWD/dataset:/workspace/dataset" \
  --entrypoint bash \
  cdgs -lc "
    source /opt/conda/etc/profile.d/conda.sh
    conda activate nerfstudio
    export TORCH_DISABLE_DYNAMO=1
    export TORCHDYNAMO_DISABLE=1
    export OMP_NUM_THREADS=12

    gs-mesh gaussians \
      --load-config /workspace/${CONFIG} \
      --output-dir /workspace/dataset/dn-splatter/export/mesh/gaussians_poisson_d13_more_detail \
      --min-opacity 0.003 \
      --down-sample-voxel 0.0005 \
      --outlier-removal True \
      --std-ratio 4.0 \
      --poisson-depth 13
  "
```

TSDF bývá rychlejší, ale může vyhladit tenké větve/listy/plody. Poisson z gaussianů často zachová víc detailu, ale může přidat blány a šum. Marching cubes je dostupný přes `gs-mesh marching`, ale vyšší `--resolution` roste kubicky a pro detailní scény bývá velmi pomalý.

## Použití existujícího COLMAPu

Pokud už máš validní model, dej ho do:

```text
dataset/colmap/sparse/0
```

a spusť:

```bash
docker run --rm --gpus all \
  -v "$PWD/dataset:/workspace/dataset" \
  -e RUN_COLMAP=never \
  cdgs
```
