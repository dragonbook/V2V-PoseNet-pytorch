from torch.utils.data import sampler
from torch.utils.data import Dataset


class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


class ChunkDataset(Dataset):
    '''
    A warpper of common datasets
    '''
    def __init__(self, data_set, num_samples, start=0):
        self.data_set = data_set
        self.num_samples = num_samples
        self.start = start
        assert(self.start + self.num_samples <= len(data_set))

    def __getitem__(self, index):
        return self.data_set[index]

    def __len__(self):
        return self.num_samples
