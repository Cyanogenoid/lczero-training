import torch
import lc0worker
from tqdm import tqdm

for _ in tqdm(range(10**8)):
    tensor = lc0worker.load_position()
    #print(tensor)
