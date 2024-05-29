import time
from tqdm import tqdm

import utils.pytorch.dataset.datapicker


def buffer_dataloader(dataloader):
    tqdm.write("Loading Data...", end="")
    start_time = time.time()
    datas = enumerate(dataloader)
    end_time = time.time()
    if end_time - start_time <= 0.01:
        time.sleep(0.5)
        tqdm.write("\r", end="")
    else:
        tqdm.write(f"\rTime to load data: {end_time - start_time:.2f}s")

    return datas