import torch
import time

device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda")

x_cpu = torch.rand(90000, 90000, device=device_cpu)
x_gpu = torch.rand(90000, 90000, device=device_gpu)

start = time.time()
y_cpu = x_cpu @ x_cpu
print("CPU:", time.time() - start)

_ = x_gpu @ x_gpu

start = time.time()
y_gpu = x_gpu @ x_gpu
torch.cuda.synchronize()
print("GPU:", time.time() - start)
