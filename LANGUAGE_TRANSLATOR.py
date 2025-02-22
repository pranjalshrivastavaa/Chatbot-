import torch
print(torch.__version__)
import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import random
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset

# Step 1: Load Spacy Tokenizers
spacy_en = spacy.load("en_core_web_sm")
spacy_fr = spacy.load("fr_core_news_sm")

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def tokenize_fr(text):
    return [tok.text.lower() for tok in spacy_fr.tokenizer(text)]

# Sample Dataset
pairs = [
    ("hello, how are you?", "bonjour, comment ça va?"),
    ("good morning", "bonjour"),
    ("i love deep learning", "j'aime l'apprentissage profond"),
]

# Build Vocabulary
def yield_tokens(data, tokenizer):
    for src, tgt in data:
        yield tokenizer(src)
        yield tokenizer(tgt)

SRC_VOCAB = build_vocab_from_iterator(yield_tokens(pairs, tokenize_en), specials=["<unk>", "<pad>", "<sos>", "<eos>"])
TGT_VOCAB = build_vocab_from_iterator(yield_tokens(pairs, tokenize_fr), specials=["<unk>", "<pad>", "<sos>", "<eos>"])
SRC_VOCAB.set_default_index(SRC_VOCAB["<unk>"])
TGT_VOCAB.set_default_index(TGT_VOCAB["<unk>"])

# Encoder Model
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        hidden = self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)).unsqueeze(0)
        return outputs, hidden, cell

# Decoder Model
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

# Training Loop
INPUT_DIM = len(SRC_VOCAB)
OUTPUT_DIM = len(TGT_VOCAB)
ENC_EMB_DIM = DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)

model = nn.Sequential(enc, dec)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train():
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        src, tgt = torch.tensor([SRC_VOCAB["hello"]]), torch.tensor([TGT_VOCAB["bonjour"]])
        enc_out, hidden, cell = enc(src)
        output, _, _ = dec(tgt, hidden, cell)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

train()