import gc
import torch

print(torch.cuda.memory_summary(device=None, abbreviated=False))

torch.cuda.empty_cache()
gc.collect()