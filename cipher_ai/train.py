import torch
import torch.nn as nn
import torch.optim as optim
from dataset import make_dataset
from utils import build_vocab
from model import Seq2SeqTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

stoi, itos = build_vocab()
VOCAB_SIZE = len(stoi)

def encode(text):
    return torch.tensor([stoi.get(ch, 0) for ch in text], dtype=torch.long)

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = [encode(x) for x in xs]
    ys = [encode(y) for y in ys]
    xs = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    ys = nn.utils.rnn.pad_sequence(ys, batch_first=True)
    return xs, ys

data = make_dataset(2000)
loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True, collate_fn=collate_fn)

model = Seq2SeqTransformer(VOCAB_SIZE).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        optimizer.zero_grad()
        out = model(src, tgt)
        out = out.view(-1, VOCAB_SIZE)
        loss = criterion(out, tgt.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "model.pth")
