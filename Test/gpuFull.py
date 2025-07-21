import torch
import time

device = torch.device("cuda")

size = 8192
a = torch.rand(size, size, device=device)
b = torch.rand(size, size, device=device)

print("Начинаю грузить GPU... Нажми Ctrl+C чтобы остановить")

i = 0
while True:
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    i += 1
    print(f"[{i}] Операция завершена за {elapsed:.4f} сек")