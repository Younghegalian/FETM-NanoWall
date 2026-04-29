# Data Layout

Local SEM data can be placed here without entering version control.

```text
data/raw/<sample_id>/<image-file>
data/configs/<sample_id>.json
data/precomputed_depth/<sample_id>.npy
```

`data/raw` and `data/precomputed_depth` are ignored by git because they may contain
large or unpublished research data. `data/configs` is intended to be small and
versionable.

Typical run:

```bash
sem-to-domain \
  --config data/configs/sample_001.json \
  --image data/raw/sample_001/sem.tif \
  --out runs/sample_001
```

For pipeline tuning before DA3 is installed or before model weights are downloaded:

```bash
sem-to-domain \
  --config data/configs/sample_001.json \
  --image data/raw/sample_001/sem.tif \
  --depth-npy data/precomputed_depth/sample_001.npy \
  --out runs/sample_001
```
