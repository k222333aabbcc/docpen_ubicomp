from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset


class CustomDataset(Dataset):
    def __init__(self, data, label1, label2=None):
        assert(len(data) == len(label1))
        if label2 != None:
            assert(len(data) == len(label2))
        self.data = data
        self.label1 = label1
        self.label2 = label2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label1 = self.label1[idx]
        if self.label2 == None:
            return data, label1
        label2 = self.label2[idx]
        return data, label1, label2
    