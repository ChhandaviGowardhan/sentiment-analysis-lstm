import torch
import torch.nn as nn
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        x = self.embedding(x)           # (batch, seq_len) → (batch, seq_len, embed_dim)
        _, (hidden, _) = self.lstm(x)   # take last hidden state
        x = hidden[-1]                  # (batch, hidden_dim)
        x = self.fc(x)                  # (batch, 1)
        return x
