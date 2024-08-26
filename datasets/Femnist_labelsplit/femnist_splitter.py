"""
Data splitter from the Femnist dataset.
Exctract femnist.zip, then run this to generate a partition of femnist for the desired amount of nodes. 

"""

import json
import os
from collections import defaultdict

import torch

from decentralizepy.datasets.Partitioner import KShardDataPartitioner

NB_CLASSES = 62


def read_file(file_path):
    """
    Read data from the given json file

    Parameters
    ----------
    file_path : str
        The file path

    Returns
    -------
    tuple
        (users, num_samples, data)

    """
    with open(file_path, "r") as inf:
        client_data = json.load(inf)
    return (
        client_data["users"],
        client_data["num_samples"],
        client_data["user_data"],
    )


def partition_data(data_dir, target_dir):
    """
    Function to read all the FEMNIST data files in the directory

    Parameters
    ----------
    data_dir : str
        Path to the folder containing the data files

    Returns
    -------

    """

    all_data = [[] for _ in range(NB_CLASSES)]

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(".json")]
    bar_size = 40
    for i, f in enumerate(files):
        filled = int(bar_size * i / len(files))
        print(
            f"Aggregating files: [{'#' * filled}{'-' * (bar_size - filled)}] {i/len(files)*100:.2f}%",
            end="\r",
        )
        file_path = os.path.join(data_dir, f)
        u, n, d = read_file(file_path)
        current_data = d[u[0]]
        for sample, label in zip(current_data["x"], current_data["y"]):
            all_data[label].append(sample)

    print("\nConverting to torch tensor")

    all_data_joint = []
    for label, label_data in enumerate(all_data):
        current_label_trainset = [
            (torch.tensor(a), torch.tensor(label)) for a in label_data
        ]
        all_data_joint.extend(current_label_trainset)

    print("Conversion OK")
    c_len = len(all_data_joint)

    e = c_len // NB_NODES
    frac = e / c_len
    sizes = [frac] * NB_NODES
    sizes[-1] += 1.0 - frac * NB_NODES

    return KShardDataPartitioner(all_data_joint, sizes=sizes, shards=2, seed=1234)

    # print("Saving data split")
    # with open(path, "w") as f:
    #     json.dump(num_samples, f)
    # return


NB_NODES = 64


def main():
    data_dir = (
        "results/datasets/Femnist_labelsplit/femnist/per_user_data/per_user_data/train/"
    )
    result_dir = (
        f"results/datasets/Femnist_labelsplit/femnist/data/train/{NB_NODES}nodes/"
    )
    print("Loading dataset")
    split = partition_data(data_dir, result_dir)

    for node in range(NB_NODES):
        path = os.path.join(result_dir, f"data_{node}.pt")
        partition = split.use(node)
        data_partition = [partition[i] for i in range(len(partition))]
        print(f"Saving for node {node}.")
        torch.save(data_partition, path)
    print("Dataset split and saved!")
    return


if __name__ == "__main__":
    main()
