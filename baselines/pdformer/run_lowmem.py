"""run_model.py with the memory-light dataset class registered (no library file is edited). Same command-line arguments."""
import runpy

import libcity.data.dataset as datasets

from pdformer_lowmem import PDFormerLowMemDataset

datasets.PDFormerDataset = PDFormerLowMemDataset
runpy.run_path("run_model.py", run_name="__main__")
