import torch
from torch.utils.data import Dataset
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        # tokenize
        tokens = text.split()
        # encode
        encoded = [
            self.word2idx.get(token, self.word2idx['<UNK>'])
            for token in tokens
        ]
        # pad
        padded = encoded[:self.max_len]
        if len(padded) < self.max_len:
            padded += [0] * (self.max_len - len(padded))
        return torch.tensor(padded, dtype=torch.long), torch.tensor(label, dtype=torch.float)