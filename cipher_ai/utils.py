import string
ALPHABET = string.ascii_lowercase + " "
PAD_TOKEN = "<PAD>"

def build_vocab():
    vocab = [PAD_TOKEN] + list(ALPHABET)
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos
