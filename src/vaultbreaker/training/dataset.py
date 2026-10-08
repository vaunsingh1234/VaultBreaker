import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
from typing import Dict, List, Tuple, Iterator

from vaultbreaker.models.unified_net import METHOD_TO_IDX

class MultiModalStegoDataset(Dataset):
    """
    Dataset storing feature vectors for Image, Audio, and Video files,
    their binary labels, and their embedding method indices.
    """
    def __init__(self, data_by_format: Dict[str, np.ndarray]):
        self.samples = []
        for m_type in ["image", "audio", "video"]:
            x_key = f"X_{m_type}"
            y_key = f"y_{m_type}"
            m_key = f"methods_{m_type}"
            r_key = f"rates_{m_type}"

            if x_key in data_by_format and len(data_by_format[x_key]) > 0:
                X = data_by_format[x_key]
                y = data_by_format[y_key]
                methods = data_by_format[m_key]
                rates = data_by_format[r_key]

                for i in range(len(X)):
                    m_str = str(methods[i])
                    m_idx = METHOD_TO_IDX.get(m_str, 0)
                    self.samples.append({
                        "x": torch.from_numpy(X[i]).float(),
                        "y": torch.tensor(y[i], dtype=torch.float32),
                        "method_idx": torch.tensor(m_idx, dtype=torch.long),
                        "rate": torch.tensor(rates[i], dtype=torch.float32),
                        "media_type": m_type
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]

class BalancedFormatBatchSampler(Sampler):
    """
    Samples batches with balanced representation across all active media types
    and balanced 50/50 clean vs stego ratio.
    """
    def __init__(self, dataset: MultiModalStegoDataset, batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group indices by (media_type, label)
        self.groups: Dict[Tuple[str, int], List[int]] = {}
        for idx, item in enumerate(dataset.samples):
            key = (item["media_type"], int(item["y"].item()))
            if key not in self.groups:
                self.groups[key] = []
            self.groups[key].append(idx)

        self.num_batches = len(dataset) // batch_size
        if self.num_batches == 0 and len(dataset) > 0:
            self.num_batches = 1

    def __iter__(self) -> Iterator[List[int]]:
        active_groups = list(self.groups.keys())
        for _ in range(self.num_batches):
            batch = []
            # Draw equally from available (format, label) groups
            while len(batch) < self.batch_size:
                g_key = active_groups[np.random.randint(len(active_groups))]
                pool = self.groups[g_key]
                idx = pool[np.random.randint(len(pool))]
                batch.append(idx)
            yield batch

    def __len__(self) -> int:
        return self.num_batches
