from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Fixed per dataset: split, the smallest target counted in MAPE, and the smallest per-sensor standard deviation.
DATASETS = {
    "gba": {"train_rate": 0.8, "val_rate": 0.1, "mape_min_flow": 10.0, "min_std": 1.0},
    "pems_bay": {"train_rate": 0.7, "val_rate": 0.1, "mape_min_flow": 1.0, "min_std": 0.1},
}


@dataclass
class DataConfig:
    path: str = "data/gba_2021.npz"
    dataset: str = "gba"
    train_rate: float = 0.8
    val_rate: float = 0.1
    seq_len: int = 12
    horizons: tuple[int, ...] = (3, 6, 12)
    mape_min_flow: float = 10.0
    min_std: float = 1.0

    @classmethod
    def from_file(cls, path: str) -> "DataConfig":
        """The configuration of the dataset a prepared file belongs to (scripts/prepare_data.py records it)."""
        with np.load(path) as z:
            name = str(z["dataset"]) if "dataset" in z.files else "gba"
        return cls(path=path, dataset=name, **DATASETS[name])


@dataclass
class GraphConfig:
    """Part A: restricted VAR, impulse responses, knockoff selection, graph assembly."""
    var_lag: int = 3
    var_ridge: float = 1e6
    stride: int = 3            # one row every 15 minutes for the knockoff statistic
    self_lags: int = 3
    shrink: float = 0.01       # ridge shrinkage of the candidate correlation matrix
    group_corr: float = 0.5    # candidates with |r| >= group_corr form one knockoff group
    fdr: float = 0.3           # knockoff+ level q
    max_sources: int = 25
    min_rows: int = 2000
    pc_shrink: float = 0.02    # shrinkage of the partial-correlation estimate used for added-edge weights
    out_dir: str = "graphs"


@dataclass
class ModelConfig:
    hidden: int = 64
    horizon_embed: int = 8
    dropout: float = 0.1
    use_gate: bool = True
    use_graph: bool = True
    two_hop: bool = False


@dataclass
class TrainConfig:
    graph_config: str = "family"   # family | g12_all | g3_all | shuffled | road | road_power | none
    seed: int = 42
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    epochs: int = 40
    warmup_epochs: int = 1
    patience: int = 6
    grad_clip: float = 5.0
    results_dir: str = "results"
    model: ModelConfig = field(default_factory=ModelConfig)

    def run_dir(self) -> Path:
        flags = "".join(f"_{f}" for f, on in (("nogate", not self.model.use_gate), ("nograph", not self.model.use_graph),
                                              ("2hop", self.model.two_hop)) if on)
        return Path(self.results_dir) / f"{self.graph_config}{flags}__seed{self.seed}"
