from collections import defaultdict
from torch.utils.data import Sampler
import numpy as np
from torch.utils.data import DataLoader

np.random.seed(42)
def group_data_by_id(dataset):
    grouped_data = defaultdict(list)
    for i in range(len(dataset)):
        _, _, group_id = dataset[i]
        grouped_data[group_id].append(i)
    return grouped_data

class GroupedBatchSampler(Sampler):
    def __init__(self, grouped_data, batch_size):
        self.grouped_data = grouped_data
        self.batch_size = batch_size
        self.batches = self._create_batches()

    def _create_batches(self):
        batches = []
        group_indices = list(self.grouped_data.keys())
        # Shuffle group indices
        np.random.shuffle(group_indices)
        # print("group_indices: ", group_indices)

        for i in range(0, len(group_indices), self.batch_size // 2):
            batch = []
            for group_idx in group_indices[i:i+self.batch_size // 2]:
                for data in self.grouped_data[group_idx]:
                    batch.append(data)
            batches.append(batch)
        
        # print(batch)
        # Shuffle batches
        np.random.shuffle(batches)
        return batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

