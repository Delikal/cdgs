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
| `STEPS` | `4000` | Počet trénovacích iterací. |
| `DOWNSCALE_FACTOR` | `2` | Hodnota předaná do `ns-train --downscale-factor`. |
| `DEPTH_ENCODER` | `vits` | Depth Anything encoder. Image obsahuje checkpointy `vits` a `vitb`. |

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
