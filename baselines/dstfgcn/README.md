# DSTFGCN baseline on PEMS-BAY

Runs the authors' DSTFGCN (https://github.com/SYLan2019/DSTFGCN) on the same PEMS-BAY data, split, road graph, metrics and
common training settings as HGP. The authors' `train.py`, `engine.py`, `model/` and `util.py` are not modified or copied here:
the notebook clones their repository and adds the files in this folder. (The repository has no licence file.)

- `dstfgcn_pems_bay_kaggle.ipynb` - Kaggle notebook (GPU): prepare data, train, HGP-style test metrics.
- `PEMS-BAY_hgp.json` - config: HGP common settings (AdamW, lr 3e-4, wd 1e-4, batch 16, 40 epochs, 1 warm-up epoch + cosine,
  patience 6, clip 5, hidden 64, dropout 0.1, seed 42, split 70/10/20); the other values are the authors' PEMS-BAY config.
- `prepare_pemsbay.py` - pems-bay.h5 + adj_mx_bay.pkl to `data/PEMS-BAY/{train,val,test}.npz` with the authors' own windowing
  function (`generate_seq`); HGP's cut points; DCRNN road graph stored as `[ids, id_to_index, adj]` (train.py reads `adj[2]`).
- `run_hgp_protocol.py` - swaps the hard-coded Adam / per-epoch MultiStepLR for AdamW and HGP's per-batch warm-up + cosine
  schedule, then runs `train.py` unchanged.
- `evaluate_hgp_metrics.py` - test MAE / RMSE / MAPE at steps 3, 6, 12, their mean and the RMSE growth, as in `hgp/metrics.py`.

Checked on a 30-sensor slice on CPU: data preparation, training (2 epochs) and evaluation run; the evaluator reproduces the
authors' own per-horizon test numbers; the optimiser is AdamW and the learning rate follows the warm-up.

Known differences from the authors' setup: 40 epochs instead of 200; hidden width 64 (authors 32); dropout 0.1 (authors 0.2);
batch 16 (authors 32); 70/10/20 split (authors 60/20/20); AdamW and warm-up + cosine instead of Adam and step decay; model selection is
the authors' (lowest validation Huber loss), HGP's is validation MAE; missing readings stay 0 in the inputs (HGP interpolates gaps up
to 30 minutes). The authors' own PEMS-BAY adjacency file in their repository is a plain matrix that their `train.py` cannot read, so the DCRNN graph is used.
Status: not yet run on the full data; seconds per epoch on a GPU are not measured.
