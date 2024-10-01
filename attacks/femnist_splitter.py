"""
Data splitter from the Femnist dataset.
Exctract femnist.zip, then run this to generate a partition of femnist for the desired amount of nodes. 

"""

import argparse
import json
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
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


def partition_data(data_dir, nb_shards):
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

    print(f"Partitionning data in {nb_shards} shards")
    return KShardDataPartitioner(
        all_data_joint, sizes=sizes, shards=nb_shards, seed=1234
    )

    # print("Saving data split")
    # with open(path, "w") as f:
    #     json.dump(num_samples, f)
    # return


NB_NODES = 64


def get_class_stats(data):
    i = 0
    try:
        classes = [0 for _ in range(NB_CLASSES)]
        for i in range(len(data)):
            classes[data[i][1].item()] += 1
        nb_classes = sum([1 if e != 0 else 0 for e in classes])
        return nb_classes
    except Exception as e:
        print(f"Failed to get class stats - {i}/{len(data)} - got error:\n{e}")
        print(data[i][1])
        return 0


def main(nb_shards):
    data_dir = (
        "attacks/datasets/Femnist_labelsplit/femnist/per_user_data/per_user_data/train/"
    )
    result_dir = f"attacks/datasets/Femnist_labelsplit/femnist/data/train/{NB_NODES}nodes_{nb_shards}shards/"
    assert os.path.exists(
        data_dir
    ), f"{data_dir} does not exist! Are you sure you are in the correct directory?"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    print("Loading dataset")
    t0 = time.time()
    split = partition_data(data_dir, nb_shards=nb_shards)
    t1 = time.time()
    print(f"Data splitting and conversion took {(t1-t0)/60:.2f} minutes")
    overall_stats = []
    for node in range(NB_NODES):
        path = os.path.join(result_dir, f"data_{node}.pt")
        partition = split.use(node)
        data_partition = [partition[i] for i in range(len(partition))]
        print(f"Saving for node {node}.")
        torch.save(data_partition, path)
        stats = get_class_stats(data_partition)
        overall_stats.append(stats)
        print(f"Node {node} got {stats} classes")
    print("Dataset split and saved!")
    t2 = time.time()
    print(f"Saving took {(t2-t1)/60:.2f} minutes")
    print(f"Overall time: {(t2-t0)/60:.2f} minutes")
    print(overall_stats)
    plt.bar([i for i in range(NB_NODES)], overall_stats)
    plt.ylabel("Number of classes")
    plt.xlabel("Node id")
    plt.axhline(NB_CLASSES / 2)
    plt.title(f"Class repartition fo {NB_NODES} nodes and {nb_shards} shards.")
    plt.savefig(os.path.join(result_dir, "class_repartition.png"))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=20)
    parser.add_argument("--nodes", type=int, default=64)

    args = parser.parse_args()
    NB_SHARDS = args.shards
    NB_NODES = args.nodes
    main(NB_SHARDS)
