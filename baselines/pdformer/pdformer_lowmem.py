"""Memory-light data loading for PDFormer. The model, loss, masks and metrics are untouched.

PDFormerDataset materialises every (input, target) window as one dense float64 array. For PEMS-BAY that is
52,093 windows x 12 steps x 325 sensors x 9 features x 8 bytes = 14.6 GB for the inputs and the same for the targets
(plus copies made while splitting and padding), which does not fit in a 30 GB machine. This subclass keeps the single
(time x sensor x feature) matrix and cuts each window out of it when a batch asks for it.

It reproduces the original pipeline exactly:
  * the same feature matrix (speed, time of day, one-hot day of week) built by the original loader methods;
  * the same chronological split (round() of the sample counts) and the same repeat-the-last-sample padding of every split;
  * the same StandardScaler, fitted on the speed channel of all training input windows (computed from per-row window
    counts instead of from the materialised windows - the same mean and standard deviation);
  * the same pattern-key clustering call on the same candidate windows.
tests/check_lowmem.py compares it with the original on a small slice.
"""
import os

import numpy as np
from torch.utils.data import DataLoader

from libcity.data.batch import Batch
from libcity.data.dataset.pdformer_dataset import PDFormerDataset
from libcity.data.list_dataset import ListDataset


class Windows:
    """Lazy list of (x, y) windows over a (T, N, F) matrix; window w covers rows w..w+in-1 (x) and w+in..w+in+out-1 (y)."""

    def __init__(self, matrix, first, count, n_in, n_out, batch_size):
        self.matrix, self.first, self.count = matrix, first, count
        self.n_in, self.n_out = n_in, n_out
        self.length = count + (batch_size - count % batch_size) % batch_size

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i < 0:
            i += self.length
        w = self.first + min(i, self.count - 1)
        x = self.matrix[w: w + self.n_in]
        y = self.matrix[w + self.n_in: w + self.n_in + self.n_out]
        return x.copy(), y.copy()


class PDFormerLowMemDataset(PDFormerDataset):

    def _feature_matrix(self):
        if isinstance(self.data_files, list):
            data_files = self.data_files.copy()
        else:
            data_files = [self.data_files].copy()
        mats = []
        for filename in data_files:
            df = self._load_dyna(filename)
            if self.load_external:
                df = self._add_external_information(df, None)
            mats.append(df)
        return np.concatenate(mats, axis=0)

    def _split_counts(self, num_samples):
        test_rate = 1 - self.train_rate - self.eval_rate
        num_test = round(num_samples * test_rate)
        num_train = round(num_samples * self.train_rate)
        num_val = num_samples - num_test - num_train
        return num_train, num_val, num_test

    def get_data(self):
        n_in, n_out = self.input_window, self.output_window
        F = self._feature_matrix()
        num_samples = F.shape[0] - n_in - n_out + 1
        num_train, num_val, num_test = self._split_counts(num_samples)
        first_train = int(num_train * (1 - self.part_train_rate))
        n_train = num_train - first_train
        self._logger.info("Dataset created (lazy windows)")
        self._logger.info("windows: train {}, eval {}, test {}, feature dim {}".format(
            n_train, num_val, num_test, F.shape[-1]))

        self.feature_dim = F.shape[-1]
        self.ext_dim = self.feature_dim - self.output_dim

        # StandardScaler of the original: mean/std of the speed channel over all training input windows
        d = self.output_dim
        rows = np.arange(first_train, num_train + n_in - 1)
        counts = np.minimum(rows, num_train - 1) - np.maximum(first_train, rows - (n_in - 1)) + 1
        speed = F[rows[0]: rows[-1] + 1, :, :d].astype(np.float64)
        total = counts.sum() * speed.shape[1] * d
        mean = (speed.sum(axis=(1, 2)) * counts).sum() / total
        var = ((speed ** 2).sum(axis=(1, 2)) * counts).sum() / total - mean ** 2
        self.scaler = self._get_scalar_from(mean, np.sqrt(var))
        self.ext_scaler = self._get_scalar(self.ext_scaler_type, F[:1, :, d:], F[:1, :, d:])
        del speed

        F[..., :d] = self.scaler.transform(F[..., :d])

        bs = self.batch_size
        train = Windows(F, first_train, n_train, n_in, n_out, bs)
        val = Windows(F, num_train, num_val, n_in, n_out, bs)
        test = Windows(F, num_samples - num_test, num_test, n_in, n_out, bs)

        def collator(items):
            batch = Batch(self.feature_name, None, None)
            for item in items:
                batch.append(item)
            batch.padding()
            return batch

        self.train_dataloader = DataLoader(ListDataset(train), batch_size=bs, num_workers=self.num_workers,
                                           collate_fn=collator, shuffle=True)
        self.eval_dataloader = DataLoader(ListDataset(val), batch_size=bs, num_workers=self.num_workers,
                                          collate_fn=collator, shuffle=True)
        self.test_dataloader = DataLoader(ListDataset(test), batch_size=bs, num_workers=self.num_workers,
                                          collate_fn=collator, shuffle=False)
        self.num_batches = len(self.train_dataloader)

        self.pattern_key_file = os.path.join(
            './libcity/cache/dataset_cache/', 'pattern_keys_{}_{}_{}_{}_{}_{}'.format(
                self.cluster_method, self.dataset, self.cand_key_days, self.s_attn_size, self.n_cluster,
                self.cluster_max_iter))
        if not os.path.exists(self.pattern_key_file + '.npy'):
            from tslearn.clustering import KShape, TimeSeriesKMeans
            k = min(self.cand_key_days * self.points_per_day, n_train)
            idx = np.arange(first_train, first_train + k)[:, None] + np.arange(self.s_attn_size)[None, :]
            cand = F[idx][..., :self.output_dim]                       # (k, s_attn_size, N, output_dim)
            pattern_cand_keys = cand.swapaxes(1, 2).reshape(-1, self.s_attn_size, self.output_dim)
            self._logger.info("Clustering...")
            if self.cluster_method == "kshape":
                km = KShape(n_clusters=self.n_cluster, max_iter=self.cluster_max_iter).fit(pattern_cand_keys)
            else:
                km = TimeSeriesKMeans(n_clusters=self.n_cluster, metric="softdtw",
                                      max_iter=self.cluster_max_iter).fit(pattern_cand_keys)
            self.pattern_keys = km.cluster_centers_
            np.save(self.pattern_key_file, self.pattern_keys)
            self._logger.info("Saved at file " + self.pattern_key_file + ".npy")
        else:
            self.pattern_keys = np.load(self.pattern_key_file + ".npy")
            self._logger.info("Loaded file " + self.pattern_key_file + ".npy")
        return self.train_dataloader, self.eval_dataloader, self.test_dataloader

    def _get_scalar_from(self, mean, std):
        from libcity.utils.normalization import StandardScaler as Std
        assert self.scaler_type == "standard", "the low-memory loader reproduces the standard scaler only"
        self._logger.info('StandardScaler mean: ' + str(mean) + ', std: ' + str(std))
        return Std(mean=mean, std=std)
