---
id: data-loaders-configs
track: ch02-python
title: "Data loaders & configs that scale"
badge: CORE
minutes: 9
prereqs: []
tags: [python, data-loading, config, pydantic, yaml, production]
xp: 45
hot2026: false
---

Imagine you shipped a training pipeline last quarter. It worked beautifully on your laptop. Then a new team member joins, changes a single hardcoded path, forgets to update a second copy buried three files down — and the overnight training run silently trains on stale data for six hours before anyone notices. No error. Just wrong results.

This is the data loader and config problem. It's not glamorous. But it is the difference between a pipeline you can trust and one that surprises you at 2 a.m.

## What "scaling" actually means here

Scaling doesn't only mean "handling a billion records." It also means:

- A second engineer can run your pipeline without a Slack thread asking where things live.
- Switching from a small dev dataset to a full production dataset is one config change, not a code edit.
- You can trace every training run back to the exact data and settings used.

That's the target. Let's get there.

## Configs first: stop hardcoding everything

The fastest way to make a pipeline fragile is to scatter magic values through your code — batch sizes, file paths, learning rates, feature lists. When something needs to change (and it always does), you hunt across ten files.

The fix is a single config object that owns all settings. Two patterns dominate production work:

**YAML + Pydantic** is the most common combo. YAML gives humans a readable place to edit values. Pydantic validates them at startup — wrong type, missing field, impossible value — before your GPU even warms up.

**dataclasses** or plain Python `@dataclass` work fine for smaller projects where you want zero extra dependencies.

:::why-prod
In production, a config validation error at startup is far cheaper than a silent wrong-value bug discovered six hours into a training run. Pydantic catches bad configs before any compute is wasted.
:::

```python {title="config.py — typed config with Pydantic v2" run=false}
# pip install pydantic pyyaml  (both free, open-source)
from pydantic import BaseModel, field_validator, PositiveInt
from pathlib import Path
import yaml

class DataConfig(BaseModel):
    train_path: Path
    val_path: Path
    batch_size: PositiveInt = 32
    num_workers: int = 4
    max_samples: int | None = None   # None = use all

    @field_validator("train_path", "val_path")
    @classmethod
    def must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Path not found: {v}")
        return v

class TrainConfig(BaseModel):
    data: DataConfig
    learning_rate: float = 3e-4
    epochs: PositiveInt = 10
    experiment_name: str = "default"

def load_config(path: str) -> TrainConfig:
    raw = yaml.safe_load(open(path))
    return TrainConfig(**raw)

# Usage:
# cfg = load_config("configs/train_dev.yaml")
# cfg.data.batch_size  -> 32 (validated, typed)
```

A matching YAML file looks like this:

```yaml {title="configs/train_dev.yaml" run=false}
data:
  train_path: "data/train.parquet"
  val_path:   "data/val.parquet"
  batch_size: 64
  num_workers: 2
learning_rate: 0.001
epochs: 5
experiment_name: "experiment-2026-q2"
```

Switch environments by pointing at a different YAML file. The code never changes.

## Data loaders that don't surprise you

A data loader's job: get data from disk (or network) into your model efficiently and reproducibly. Two failure modes are common:

1. **It reads the entire dataset into RAM** — fine for toy data, fatal at scale.
2. **It's non-deterministic** — shuffle without a fixed seed means you can't reproduce training results.

Python's `pathlib` plus generators handle the first. A fixed seed handles the second.

:::table {title="Loader patterns by use case"}
| Use case | Pattern | Key benefit |
|---|---|---|
| CSV / Parquet, fits in RAM | `pandas.read_parquet` + slice | Simple, fast for small data |
| Large files | Generator / `yield` chunks | Constant memory usage |
| Images / multimodal | PyTorch `Dataset` + `DataLoader` | Prefetch, multi-worker |
| Streaming (Kafka, S3 live) | `iter()` + batching | No disk required |
:::

```python {title="chunked_loader.py — memory-safe generator" run=false}
# No extra installs needed beyond pandas + pyarrow
# pip install pandas pyarrow  (free, open-source)
import pandas as pd
from pathlib import Path
from typing import Iterator

def load_in_chunks(
    path: Path,
    chunk_size: int = 1_000,
    seed: int = 42,
) -> Iterator[pd.DataFrame]:
    """
    Yields shuffled chunks from a parquet file.
    Memory stays flat no matter how big the file is.
    """
    df = pd.read_parquet(path)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start : start + chunk_size]

# Usage:
# for batch_df in load_in_chunks(cfg.data.train_path, cfg.data.batch_size):
#     train_on(batch_df)
```

Seeding the shuffle here means two engineers running the same config get the same training batches. That's reproducibility by default, not by accident.

## Wiring config to loader

The power move is passing the config object directly into loader functions. No free-floating variables, no globals.

```python {title="pipeline.py — config-driven entry point" run=false}
from config import load_config
from chunked_loader import load_in_chunks

def run(config_path: str) -> None:
    cfg = load_config(config_path)          # validated at this line
    for batch in load_in_chunks(
        path=cfg.data.train_path,
        chunk_size=cfg.data.batch_size,
    ):
        # your training step here
        pass

if __name__ == "__main__":
    import sys
    run(sys.argv[1])   # python pipeline.py configs/train_dev.yaml
```

One argument. One entry point. Any engineer on your team — or you, six months from now — can run this without reading the source.

:::gotcha
Never use `os.environ` or hardcoded strings as fallbacks inside the loader itself (e.g. `path = os.environ.get("TRAIN_PATH", "/my/local/path")`). It feels convenient and causes the exact silent-wrong-data bug described at the top. Put all defaults in the Pydantic model; that's the contract.
:::

:::interview-line
"I keep all magic values in a typed Pydantic config validated at startup — if the config is wrong, the job fails immediately with a clear error instead of silently training on bad data."
:::

:::qa {q="Why use Pydantic for config instead of just loading a YAML dict?"}
Pydantic validates types and constraints before any computation starts. A raw dict lets bad values — wrong path, negative batch size, missing key — pass silently until they cause a cryptic runtime error mid-training. Pydantic surfaces config mistakes in a single clear message at startup, on line one.
:::

:::qa {q="How do you make a data loader reproducible across machines?"}
Fix the random seed at the loader level and pass it through the config object so it is logged alongside every run. PyTorch DataLoader has a worker_init_fn and generator argument for this. The seed lives in config, not scattered in code.
:::

:::drill {type="mcq" q="A teammate reports that re-running the same training script gives different validation metrics each time. What is the most likely root cause?"}
- [ ] The learning rate is too high
- [ ] Pydantic validation is disabled
- [x] The data shuffle is not seeded, so batches differ each run
- [ ] The YAML config file is being cached by the OS
:::

:::drill {type="mcq" q="You have a 200 GB Parquet dataset. Which loading approach keeps memory usage flat?"}
- [ ] `pd.read_parquet(path)` followed by `.sample()`
- [ ] Reading the file with `open()` line by line
- [x] A generator that yields fixed-size chunks, seeded before iteration
- [ ] Loading into a NumPy array and slicing during training
:::

:::key-takeaway
One validated config object owns every setting; loaders receive that object directly. Fix the shuffle seed. Then config bugs and data surprises announce themselves immediately — not six hours into a training run.
:::
