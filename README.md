# HGP: Horizon-Specific Causal Graphs for Multi-Horizon Traffic Forecasting

HGP builds a separate directed dependency graph for every forecast horizon and uses each graph only when predicting
that horizon. It has two parts.

**Part A — horizon-specific graph generation.** A road-restricted ridge VAR gives horizon-*h* impulse responses
Ψ_h. For every sensor, candidate sources (sensors within two road hops) are tested with group knockoffs on the
*h*-step-ahead relation; the knockoff+ threshold keeps the selected edges' false discovery rate at a chosen level
*q*. Selected edges are weighted by the standardised VAR effect where the VAR supports them and by a partial
correlation otherwise. Output: one graph G_h per horizon.

**Part B — HICP (horizon-indexed causal propagation).** A per-node GRU encodes every sensor's history. For each
horizon, one directed propagation step over G_h produces a spatial representation, a gate conditioned on a learned
horizon embedding mixes it with the temporal one, and a horizon-specific head emits the forecast. All horizons are
produced in one forward pass; only the adjacency tensors differ between experimental configurations, so every
configuration has the same parameter count.

The accompanying specification (`HGP_architecture.pdf`) gives the equations; the code follows them one to one.

## Installation

```
pip install -r requirements.txt
```

Python ≥ 3.10, PyTorch ≥ 2.0. A GPU is needed for full-scale training (2,056 sensors); the graph generation runs on CPU.

## Data

Two datasets are supported; `scripts/prepare_data.py` writes one `.npz` per dataset and records which one it is, which
selects the fixed split and metric settings (`hgp/config.py`).

| dataset | quantity | sensors | split (train / val / test) | MAPE counts targets ≥ |
|---|---|---|---|---|
| LargeST-GBA 2021 (Liu et al., NeurIPS 2023) | flow, vehicles / 5 min | 2,056 after the duplicate filter | 80 / 10 / 10 | 10 |
| PEMS-BAY (Li et al., ICLR 2018) | speed, mph | 325 | 70 / 10 / 20 | 1 |

```
python scripts/prepare_data.py --dataset gba      --traffic gba_his_raw_2021.h5 --adjacency gba_rn_adj.npy --out data/gba_2021.npz
python scripts/prepare_data.py --dataset pems_bay --traffic pems-bay.h5         --adjacency adj_mx_bay.pkl --out data/pems_bay.npz
```

Graphs, scalers and model selection use training and validation rows only; the test period is scored once per run.
Missing values are never targets: PEMS-BAY zeros are treated as missing, and only the model inputs have short gaps
(at most 30 minutes) filled. The script prints the candidate-pool size; see its docstring for the GBA sensor filter.

## Part A: build the graphs

```
python scripts/build_graphs.py --data data/pems_bay.npz --horizons 3 6 12 --out graphs
```

Writes `graphs/G_h{3,6,12}.npz` (HGP graphs), `graphs/VAR_h{h}.npz` (VAR-only reference graphs) and the knockoff
statistics of every target. About 30 minutes per horizon for 2,056 sensors on one CPU core; a minute or two for PEMS-BAY.

## Hyperparameters

The training protocol is fixed in `hgp/config.py` and is the same for every configuration and dataset: AdamW, learning
rate 3·10⁻⁴ with one warm-up epoch and cosine decay, weight decay 10⁻⁴, gradient clipping at 5, batch size 16, at most 40
epochs with early stopping (patience 6) on the validation MAE averaged over horizons. Only the method-specific
settings are exposed as options:

| option | script | default | meaning |
|---|---|---|---|
| `--fdr` | `build_graphs.py` | 0.3 | knockoff+ level q |
| `--group-corr` | `build_graphs.py` | 0.5 | correlation above which candidates share a knockoff group |
| `--max-sources` | `build_graphs.py` | 25 | sources per target |
| `--var-lag`, `--var-ridge` | `build_graphs.py` | 3, 10⁶ | road-restricted VAR (the ridge scales with the data) |
| `--hidden`, `--dropout` | `train.py` | 64, 0.1 | temporal encoder width and dropout |

Tune them on the validation split only.

## Part B: train HICP

```
python scripts/train.py --config family  --seed 42     # G_3 / G_6 / G_12 in their own branches
python scripts/train.py --config g12_all --seed 42     # the same long-horizon graph in every branch
python scripts/train.py --config none    --seed 42     # no graph: the per-node floor
```

| `--config` | graph in branch h = 3 / 6 / 12 | question it answers |
|---|---|---|
| `family` | G_3 / G_6 / G_12 | the proposed model |
| `g12_all` | G_12 / G_12 / G_12 | does horizon matching add to long reach? |
| `g3_all` | G_3 / G_3 / G_3 | short reach everywhere |
| `shuffled` | a derangement of the family | does the graph–horizon pairing matter? |
| `road` | road adjacency | physical graph |
| `road_power` | road reach within 1 / 2 / 4 hops | hop distance without dynamics |
| `none` | – | per-node floor |

`--no-gate` and `--two-hop` give the gate and depth ablations. Results go to `results/<config>__seed<seed>/`:
`results.json` (per-horizon MAE, RMSE, masked MAPE on vehicles / 5 min; the mean gate value per horizon),
`history.csv`, `checkpoints/best.pt` and `test_predictions_h{h}.npz`.

## Kaggle

`notebooks/hgp_pems_bay_kaggle.ipynb` runs the whole pipeline in one session: data preparation, graph building (once per
setting) and every (configuration, seed) pair listed in its settings cell, followed by the results tables. Add the
PEMS-BAY files and this repository (as a Kaggle dataset, or via `REPO_URL`) as inputs and select a GPU.

## Summarising runs

```
python scripts/summarise.py --results results --metric rmse
```

prints, per configuration and horizon, the mean ± standard deviation over seeds and the gate values, followed by the
paired RMSE ratio `family / g12_all` with a 95% bootstrap interval over sensors for every seed. The pre-registered
decision rule: horizon-indexed propagation is claimed to help only if that ratio is below 1 at h = 6 and h = 12 in
every seed, with intervals excluding 1 in the majority of seeds.

## Reproducibility

Seeds fix initialisation, data order and the knockoff sampling. Runs on different GPU types can differ at the level
of a different seed; compare configurations run on the same hardware, over at least three seeds.

## Tests

```
python -m pytest tests -q
```

`tests/test_train.py` runs the whole pipeline (graph build, three configurations, two seeds, summary) on a synthetic dataset in about two minutes.

## Citation

Please cite the thesis / paper associated with this repository (reference to be added on publication).
# HGP
