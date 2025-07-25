import random
import string

ALPHABET = string.ascii_lowercase + " "

def caesar_cipher(text, shift):
    result = ""
    for ch in text.lower():
        if ch in ALPHABET:
            i = (ALPHABET.index(ch) + shift) % len(ALPHABET)
            result += ALPHABET[i]
        else:
            result += ch
    return result

def substitution_cipher(text):
    shuffled = list(ALPHABET)
    random.shuffle(shuffled)
    mapping = {a:b for a,b in zip(ALPHABET, shuffled)}
    return "".join(mapping.get(ch, ch) for ch in text.lower())

def random_cipher(text):
    if random.random() < 0.5:
        return caesar_cipher(text, random.randint(1, 10))
    return substitution_cipher(text)
