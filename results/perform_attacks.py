import os
from typing import Optional

import pandas as pd
import torch
import torchvision

from decentralizepy.datasets.CIFAR10 import CIFAR10
from decentralizepy.datasets.Dataset import Dataset
from decentralizepy.datasets.Partitioner import DataPartitioner, KShardDataPartitioner


def load_CIFAR10():
    """Generates and partitions the CIFAR10 dataset in the same way as the experiment does."""
    print("Loading CIFAR10 dataset")
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="datasets/cifar10", train=False, download=True, transform=transform
    )
    return trainset


POSSIBLE_DATASETS = {"CIFAR10": (load_CIFAR10, 10)}


# Problem: this function is built upon the loading function of CIFAR10 - no "general" function that can be easily reused can be found.
def load_dataset_partitioner(
    dataset_name: str,
    total_agents: int,
    seed: int,
    shards: int,
    sizes: Optional[list[float]] = None,
) -> KShardDataPartitioner:
    """Loads a data partitioner in an identical way as the experiments do

    Args:
        dataset_name (str): The name of the dataset: CIFAR10
        total_agents (int): The number of agents for the partition
        seed (int): The random seed used in the experiment
        shards (int): The number of shards for each agent
        sizes (float array, optional): A list of data fraction for each agent . Defaults to None, corresponding to an equal distribution.

    Raises:
        ValueError: When the dataset name is not a currently implemented dataset to be loaded.

    Returns:
        decentralizepy.datasets.KShardDataPartitioner: A data partitioner
    """
    if dataset_name not in POSSIBLE_DATASETS:
        raise ValueError(
            f"{dataset_name} is not in the list of possible datasets: {list(POSSIBLE_DATASETS)}"
        )
    loader, num_classes = POSSIBLE_DATASETS[dataset_name]
    trainset = loader()
    c_len = len(trainset)
    if sizes is None:
        # Equal distribution of data among processes, from CIFAR10 and not a simplified function.
        # Speak with Rishi about having a separate functino that performs this?
        e = c_len // total_agents
        frac = e / c_len
        sizes = [frac] * total_agents
        sizes[-1] += 1.0 - frac * total_agents
        print(f"Size fractions: {sizes}")

    train_data: dict = {key: [] for key in range(num_classes)}
    for x, y in trainset:
        train_data[y].append(x)
    all_trainset = []
    for y, x in train_data.items():
        all_trainset.extend([(a, y) for a in x])
    partitioner = KShardDataPartitioner(all_trainset, sizes, shards=shards, seed=seed)

    return partitioner


def get_dataset_stats(dataset, nb_classes: int):
    """Logs the data repartition for all classes of a dataset


    Args:
        dataset (decentralizepy.datasets.Partition): A dataset partition for a local agent
        nb_classes (int): The number of classes for the dataset

    Returns:
        int array: An array of size nb_classes corresponding to the number of element for each class
    """
    classes = [0 for _ in range(nb_classes)]
    for _, target in dataset:
        classes[target] += 1
    return classes


def get_model_attributes(name, path):
    parsed_model = name.split("_")
    iteration = int(parsed_model[1][2:])
    agent = int(parsed_model[2])
    target = int(parsed_model[3][2:-4])

    res = pd.DataFrame(
        {
            "file": os.path.join(path, name),
            "iteration": iteration,
            "agent": agent,
            "target": target,
        },
        index=[0],
    )
    return res


def list_models_path(experiment_path, machine_id, agent_id):
    models_list = pd.DataFrame({})
    for file in sorted(
        os.listdir(
            os.path.join(
                experiment_path,
                f"machine{machine_id}",
                "attacked_model",
                f"machine{machine_id}",
                f"{agent_id}",
            )
        )
    ):
        file_attributes = get_model_attributes(file, experiment_path)
        models_list = pd.concat([models_list, file_attributes])
    return models_list


if __name__ == "__main__":
    DATASET = "CIFAR10"
    NB_CLASSES = POSSIBLE_DATASETS[DATASET][1]
    NB_AGENTS = 128
    NB_MACHINES = 8
    train_partitioner = load_dataset_partitioner(DATASET, NB_AGENTS, 90, 2)
    for agent in range(NB_AGENTS):
        train_data_current_agent = train_partitioner.use(agent)
        agent_classes_trainset = get_dataset_stats(train_data_current_agent, NB_CLASSES)
        print(f"Classes for agent {agent}: {agent_classes_trainset}")

    EXPERIMENT_DIR = "results/my_results/icml_experiments/cifar10/2067267_nonoise_static_128nodes_1avgsteps_batch32_lr0.05_3rounds"

    models_df = pd.DataFrame({})
    for agent_uid in range(NB_AGENTS):
        current_machine_id = agent_uid // (NB_AGENTS // NB_MACHINES)
        agent_id = agent_uid % (NB_AGENTS // NB_MACHINES)
        models_df = pd.concat(
            [models_df, list_models_path(EXPERIMENT_DIR, current_machine_id, agent_id)]
        )
    print(models_df)
