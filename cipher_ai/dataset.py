from cipher import random_cipher
import random

def load_texts():
    phrases = [
        "hello world", "artificial intelligence",
        "deep learning is cool", "machine learning is fun",
        "pytorch makes life easier"
    ]
    return phrases

def make_dataset(n=5000):
    data = []
    texts = load_texts()
    for _ in range(n):
        text = random.choice(texts)
        ciphered = random_cipher(text)
        data.append((ciphered, text))
    return data
