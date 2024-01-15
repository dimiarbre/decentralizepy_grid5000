import os
import time
from typing import Optional

import pandas as pd
import torchvision
from localconfig import LocalConfig

from decentralizepy.datasets.CIFAR10 import LeNet
from decentralizepy.datasets.Partitioner import DataPartitioner, KShardDataPartitioner


def read_ini(file_path: str) -> LocalConfig:
    """Function to load the dict configuration file.

    Args:
        file_path (str): The path to the config file

    Returns:
        LocalConfig: The loaded configuration.
    """
    config = LocalConfig(file_path)
    for section in config:
        print("Section: ", section)
        for key, value in config.items(section):
            print((key, value))
    print(dict(config.items("DATASET")))
    return config


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
        root="datasets/cifar10", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="datasets/cifar10", train=False, download=True, transform=transform
    )
    return trainset, testset


POSSIBLE_DATASETS = {
    "CIFAR10": (
        load_CIFAR10,
        10,
        LeNet,
    )
}


# Problem: this function is built upon the loading function of CIFAR10 - no "general" function that can be easily reused can be found.
def load_dataset_partitioner(
    dataset_name: str,
    total_agents: int,
    seed: int,
    shards: int,
    sizes: Optional[list[float]] = None,
):
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
        testset: The test dataset
    """
    if dataset_name not in POSSIBLE_DATASETS:
        raise ValueError(
            f"{dataset_name} is not in the list of possible datasets: {list(POSSIBLE_DATASETS)}"
        )
    loader, num_classes, _ = POSSIBLE_DATASETS[dataset_name]
    trainset, testset = loader()
    c_len = len(trainset)
    if sizes is None:
        # Equal distribution of data among processes, from CIFAR10 and not a simplified function.
        # Speak with Rishi about having a separate functino that performs this?
        e = c_len // total_agents
        frac = e / c_len
        sizes = [frac] * total_agents
        sizes[-1] += 1.0 - frac * total_agents
        # print(f"Size fractions: {sizes}")

    train_data: dict = {key: [] for key in range(num_classes)}
    for x, y in trainset:
        train_data[y].append(x)
    all_trainset = []
    for y, x in train_data.items():
        all_trainset.extend([(a, y) for a in x])
    partitioner = KShardDataPartitioner(all_trainset, sizes, shards=shards, seed=seed)

    return partitioner, testset


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


def get_dataset_stats_batch(dataset, nb_classes: int) -> list[int]:
    classes = [0 for _ in range(nb_classes)]
    for _, target_batches in dataset:
        for target in target_batches:
            classes[target] += 1
    return classes


def get_model_attributes(name, path):
    parsed_model = name.split("_")
    iteration = int(parsed_model[1][2:])
    agent = int(parsed_model[2])
    target = int(parsed_model[3][2:-4])  # Ignore the `.npy`

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


def list_models_path(
    experiment_path: str, machine_id: int, agent_id: int
) -> pd.DataFrame:
    """Generates dataframe for a given agent and machine id

    Args:
        experiment_path (str): The directory of the experiment
        machine_id (int): The id of the machine for the agent
        agent_id (int): The id of the agent on the machine

    Returns:
        pd.Dataframe: A dataframe containing a column for the model file,
            the iteration, the agent uid and the target agent uid.
    """
    models_list = pd.DataFrame({})
    agent_models_directory = os.path.join(
        experiment_path,
        f"machine{machine_id}",
        "attacked_model",
        f"machine{machine_id}",
        f"{agent_id}",
    )
    for file in sorted(os.listdir(agent_models_directory)):
        file_attributes = get_model_attributes(file, agent_models_directory)
        models_list = pd.concat([models_list, file_attributes])
    return models_list


def get_all_models_properties(
    experiment_dir: str, nb_agents: int, nb_machines: int
) -> pd.DataFrame:
    """Generates a dataframe with all models path and properties (agent, iteration, target agent)

    Args:
        experiment_dir (str): The path to the experiment
        nb_agents (int): The total number of agent for the experiment
        nb_machines (int): The total number of machines for the experiment

    Returns:
        pd.Dataframe: A dataframe containing a column for the model file,
            the iteration, the agent uid and the target agent uid.
    """
    models_df = pd.DataFrame({})
    for agent_uid in range(nb_agents):
        current_machine_id = agent_uid // (nb_agents // nb_machines)
        agent_id = agent_uid % (nb_agents // nb_machines)
        models_df = pd.concat(
            [models_df, list_models_path(experiment_dir, current_machine_id, agent_id)]
        )
    return models_df


def get_all_experiments_properties(
    all_experiments_dir: str, nb_agents: int, nb_machines: int
) -> pd.DataFrame:
    """Generates a dataframe with all models path and properties (agent, iteration, target agent)
    for all the experiments in a folder.

    Args:
        all_experiments_dir (str): The path to the experiments
        nb_agents (int): The total number of agent for the experiments
        nb_machines (int): The total number of machines for the experiments

    Returns:
        pd.Dataframe: A dataframe containing a column for the model file,
            the iteration, the agent uid and the target agent uid, unified for all the experiments.
    """
    experiment_wide_df = pd.DataFrame({})

    for experiment_name in os.listdir(all_experiments_dir):
        experiment_path = os.path.join(all_experiments_dir, experiment_name)
        current_experiment_df = get_all_models_properties(
            experiment_path, nb_agents=nb_agents, nb_machines=nb_machines
        )
        current_experiment_df["experiment_name"] = experiment_name
        experiment_wide_df = pd.concat([experiment_wide_df, current_experiment_df])

    return experiment_wide_df


def main():
    DATASET = "CIFAR10"
    NB_CLASSES = POSSIBLE_DATASETS[DATASET][1]
    MODEL = POSSIBLE_DATASETS[DATASET][2]
    NB_AGENTS = 128
    NB_MACHINES = 8
    train_partitioner, test_data = load_dataset_partitioner(DATASET, NB_AGENTS, 90, 2)
    for agent in range(NB_AGENTS):
        train_data_current_agent = train_partitioner.use(agent)
        agent_classes_trainset = get_dataset_stats(train_data_current_agent, NB_CLASSES)
        print(f"Classes for agent {agent}: {agent_classes_trainset}")

    EXPERIMENT_DIR = "results/my_results/icml_experiments/cifar10/2067277_nonoise_dynamic_128nodes_1avgsteps_batch32_lr0.05_3rounds"

    all_models_df = get_all_models_properties(EXPERIMENT_DIR, NB_AGENTS, NB_MACHINES)
    print(all_models_df)

    EXPERIMENTS_DIR = "results/my_results/icml_experiments/cifar10/"

    t0 = time.time()
    all_experiments_df = get_all_experiments_properties(
        EXPERIMENTS_DIR, NB_AGENTS, NB_MACHINES
    )
    t1 = time.time()
    print(all_experiments_df)
    print(f"All models paths retrieved in {t1-t0:.2f}s")


if __name__ == "__main__":
    main()
