"""Script to test the loading time of the dataloader
for different infant ages vs adult vision"""

import torch as t
from data import CustomDataset
from matplotlib import pyplot as plt
import time

train_dataset = CustomDataset(data_type="jpeg", mode="train")
batch_size = 100
print("Batch size:", batch_size)
for infant_age in range(4):
    train_dataset.infant_age = infant_age
    if infant_age < 3:
        print(
            f"Vision for {train_dataset.infant_age} month old infant:",
        )
    else:
        print("Adult Vision:")

    train_dl = t.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=1,
    )
    loading_times = []
    loading_start_time = time.time()
    for i, (x, y) in enumerate(train_dl):
        current_time = time.time()
        loading_times.append(current_time - loading_start_time)
        print(f"Batch {i} took {loading_times[-1]} seconds to load")
        if i == 3:
            break
        loading_start_time = time.time()
    avg_loading_time = sum(loading_times) / len(loading_times)
    if infant_age < 3:
        print(
            f"Average loading time for {train_dataset.infant_age} months:",
            avg_loading_time,
        )
        plt.figure()
        plt.plot(loading_times)
        plt.axhline(y=avg_loading_time, color="r", linestyle="--")
        plt.title(f"Infant age: {train_dataset.infant_age} months")
        plt.xlabel("Batch")
        plt.ylabel("Loading time (s)")
        plt.savefig(f"infant_age_{train_dataset.infant_age}_loading_time.png")
    else:
        print("Average loading time for adult vision:", avg_loading_time)
        plt.figure()
        plt.plot(loading_times)
        plt.axhline(y=avg_loading_time, color="r", linestyle="--")
        plt.title("Adult vision")
        plt.xlabel("Batch")
        plt.ylabel("Loading time (s)")
        plt.savefig("adult_loading_time.png")
