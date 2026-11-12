# PDFormer baseline on PEMS-BAY

Runs the authors' PDFormer (https://github.com/BUAABIGSCity/PDFormer, MIT) on the same PEMS-BAY data, split,
metrics and common training settings as HGP. The model code is not modified or copied here: the notebook clones the
authors' repository and adds the files in this folder.

- `pdformer_pems_bay_kaggle.ipynb` - Kaggle notebook (`STAGE="caches"` on CPU, then `STAGE="train"` on GPU).
- `PEMSBAY_hgp.json` - config: HGP common settings (AdamW, lr 3e-4, wd 1e-4, batch 16, 40 epochs, warm-up 1,
  patience 6, clip 5, hidden 64, dropout 0.1, seed 42, split 70/10/20); other values from the paper or the code.
- `prepare_pemsbay.py` - pems-bay.h5 + adj_mx_bay.pkl to LibCity atomic files.
- `pdformer_lowmem.py`, `run_lowmem.py` - same windows, split, scaler and clustering as the original loader, cut lazily
  from one matrix (the original needs about 30 GB). `tests/check_lowmem.py` compares them on a slice (max diff 7e-14).
- `precompute_dtw.py` - parallel DTW (identical values); `precompute_keys.py` - KShape pattern keys.

Known differences from the published setup: 40 epochs instead of 200; PEMS-BAY structure settings (`far_mask_delta`,
`huber_delta`, `cand_key_days`, layer-norm type) taken from the PeMS04 config; `step_size` derived (2279 = batches
per epoch); dropout 0.1 (the published runs used 0). Status: not yet run on Kaggle.
