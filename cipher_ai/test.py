import torch
from utils import build_vocab
from model import Seq2SeqTransformer
from cipher import random_cipher

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

stoi, itos = build_vocab()
model = Seq2SeqTransformer(len(stoi)).to(DEVICE)
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model.eval()

def encode(text): return torch.tensor([stoi.get(ch, 0) for ch in text], dtype=torch.long).unsqueeze(0)
def decode(t): return "".join(itos[i.item()] for i in t[0])

while True:
    msg = input("Введите сообщение: ")
    ciphered = random_cipher(msg)
    print("Шифр:", ciphered)

    src = encode(ciphered).to(DEVICE)
    tgt = torch.zeros_like(src).to(DEVICE)

    with torch.no_grad():
        out = model(src, tgt).argmax(-1)
    print("ИИ расшифровал:", decode(out))
