import json
import os
import time
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from localconfig import LocalConfig
from torch.utils.data import DataLoader

from decentralizepy.datasets.CIFAR10 import LeNet
from decentralizepy.datasets.Data import Data
from decentralizepy.datasets.Femnist import CNN, RNET, Femnist
from decentralizepy.datasets.Partitioner import (
    DataPartitioner,
    KShardDataPartitioner,
    SimpleDataPartitioner,
)

# Sort this by longest computation time first to have a better scheduling policy.
ALL_ATTACKS = [
    "linkability",
    "threshold+biasedthreshold",
    "biasedthreshold",
    "threshold",
]


def read_ini(file_path: str, verbose=False) -> LocalConfig:
    """Function to load the dict configuration file.

    Args:
        file_path (str): The path to the config file

    Returns:
        LocalConfig: The loaded configuration.
    """
    config = LocalConfig(file_path)
    if verbose:
        for section in config:
            print("Section: ", section)
            for key, value in config.items(section):
                print((key, value, type(value)))
        print(dict(config.items("DATASET")))
    return config


def safe_load_int(config: LocalConfig, section: str, parameter: str):
    value = config.get(section, parameter)
    if not isinstance(value, int):
        raise ValueError(
            f"Invalid value for parameter {parameter}: expected int, got {value}"
        )
    return value


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


def femnist_read_file(file_path):
    with open(file_path, "r") as inf:
        client_data = json.load(inf)
    return (
        client_data["users"],
        client_data["num_samples"],
        client_data["user_data"],
    )


def femnist_read_dir(data_dir) -> tuple[list, list, dict]:
    """
    Function to read all the Femnist data files in the directory

    Parameters
    ----------
    data_dir : str
        Path to the folder containing the data files

    Returns
    -------
    3-tuple
        A tuple containing list of clients, number of samples per client,
        and the data items per client

    """
    clients = []
    num_samples = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(".json")]
    for f in files:
        file_path = os.path.join(data_dir, f)
        u, n, d = femnist_read_file(file_path)
        clients.extend(u)
        num_samples.extend(n)
        data.update(d)
    return clients, num_samples, data


class FemnistPartitionerWrapper(DataPartitioner):
    """
    This class is a wrapper of the Femnist Datapartitioner that is there to match the CIFAR10 syntax.
    It is basically a list, with a .use() method that is the same as .__get__()
    """

    def __init__(self, data, sizes=None, seed=1234):
        self.data = data

    def use(self, rank):
        return self.data[rank]


def load_Femnist_Testset(femnist_test_dir):
    # Load the test set
    _, _, d = femnist_read_dir(femnist_test_dir)
    test_x = []
    test_y = []
    for test_data in d.values():
        for x in test_data["x"]:
            test_x.append(x)
        for y in test_data["y"]:
            test_y.append(y)
    test_x = (
        np.array(test_x, dtype=np.dtype("float32"))
        .reshape(-1, 28, 28, 1)
        .transpose(0, 3, 1, 2)
    )
    test_y = np.array(test_y, dtype=np.dtype("int64")).reshape(-1)
    assert test_x.shape[0] == test_y.shape[0]
    assert test_x.shape[0] > 0

    return Data(test_x, test_y)


def load_Femnist(
    nb_nodes,
    sizes,
    random_seed,
    femnist_train_dir="datasets/Femnist/femnist/per_user_data/per_user_data/train",
    femnist_test_dir="datasets/Femnist/femnist/data/test/test",
    debug=False,
):
    """Generates and partitions the CIFAR10 dataset in the same way as the experiment does."""
    print("Loading Femnist dataset")
    files = os.listdir(femnist_train_dir)
    files = [f for f in files if f.endswith(".json")]
    files.sort()

    c_len = len(files)

    if sizes == None:  # Equal distribution of data among processes
        e = c_len // nb_nodes
        frac = e / c_len
        sizes = [frac] * nb_nodes
        sizes[-1] += 1.0 - frac * nb_nodes

    # Load all the trainsets
    files_partitioner = DataPartitioner(files, sizes, seed=random_seed)
    all_trainsets = []
    for uid in range(0, nb_nodes):
        if debug:
            print(f"Loading data for client {uid}.")
        my_clients = files_partitioner.use(uid)
        my_train_data = {"x": [], "y": []}
        node_clients = []
        node_num_samples = []
        if debug:
            print(f"{uid} - Client Length: {c_len}")
            print(f"{uid} - My_clients_len: {len(my_clients)}")
        for cur_file in my_clients:

            clients, _, train_data = femnist_read_file(
                os.path.join(femnist_train_dir, cur_file)
            )
            for cur_client in clients:
                node_clients.append(cur_client)
                current_client_data = train_data[cur_client]
                # f, axarr = plt.subplots(10, 1)
                # for i in range(10):
                #     to_plot = (
                #         np.array(current_client_data["x"][i])
                #         .reshape(-1, 28, 28, 1)
                #         .transpose()
                #     )
                #     axarr[i].imshow(to_plot[0])
                # plt.show()

                my_train_data["x"].extend(current_client_data["x"])
                my_train_data["y"].extend(current_client_data["y"])
                node_num_samples.append(len(train_data[cur_client]["y"]))
        node_train_x = (
            np.array(my_train_data["x"], dtype=np.dtype("float32"))
            .reshape(-1, 28, 28, 1)
            .transpose(0, 3, 1, 2)
        )
        node_train_y = np.array(my_train_data["y"], dtype=np.dtype("int64")).reshape(-1)
        assert node_train_x.shape[0] == node_train_y.shape[0]
        assert node_train_x.shape[0] > 0
        node_data = Data(node_train_x, node_train_y)
        all_trainsets.append(node_data)

    #     if debug:
    #         # subplot(r,c) provide the no. of rows and columns
    #         f, axarr = plt.subplots(10, 1)
    #         for i in range(10):
    #             to_plot = node_data.x[i][0]
    #             axarr[i].imshow(to_plot)
    # if debug:
    #     plt.show()

    testset = load_Femnist_Testset(femnist_test_dir=femnist_test_dir)

    return FemnistPartitionerWrapper(all_trainsets, [], 0), testset


def load_Femnist_labelsplit(
    nb_nodes,
    sizes,
    random_seed,
    femnist_train_dir="datasets/Femnist_labelsplit/femnist/data/train/64nodes_10shards",  # TODO: fix this hack
    femnist_test_dir="datasets/Femnist_labelsplit/femnist/data/test/test",
    debug=False,
):
    all_data = []
    for node in range(nb_nodes):
        filename = f"data_{node}.pt"
        if debug:
            print(f"Loading FEMNISTLabelSplit {filename} for node {node}.")
        node_file = os.path.join(femnist_train_dir, filename)
        node_data = torch.load(node_file)
        temp = np.array(
            [data[0] for data in node_data],
            dtype=np.dtype("float32"),
        )
        node_data_reshaped_x = temp.reshape(-1, 28, 28, 1).transpose(0, 3, 1, 2)
        node_data_reshaped_y = np.array(
            [data[1] for data in node_data], dtype=np.dtype("int64")
        ).reshape(-1)
        node_data_reshaped = Data(node_data_reshaped_x, node_data_reshaped_y)
        all_data.append(node_data_reshaped)

        if debug:
            # subplot(r,c) provide the no. of rows and columns
            f, axarr = plt.subplots(10, 1)
            for i in range(10):
                to_plot = node_data_reshaped.x[i][0]
                axarr[i].imshow(to_plot)
            f.savefig(f"assets/temp_images/{node}.png")
            plt.close()

    testset = load_Femnist_Testset(femnist_test_dir=femnist_test_dir)

    return FemnistPartitionerWrapper(all_data), testset


POSSIBLE_DATASETS = {
    "CIFAR10": (
        load_CIFAR10,
        10,
    ),
    "Femnist": (
        load_Femnist,
        62,
    ),
    "FemnistLabelSplit": (
        load_Femnist_labelsplit,
        62,
    ),
}

POSSIBLE_MODELS = {
    "RNET": RNET,
    "LeNet": LeNet,
    "CNN": CNN,
}


def load_dataset_partitioner(
    dataset_name: str,
    total_agents: int,
    seed: int,
    shards: int | None,
    sizes: Optional[list[float]] = None,
    debug=False,
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
    loader, num_classes = POSSIBLE_DATASETS[dataset_name]
    if dataset_name == "Femnist":
        # Because Femnist is handled by clients, it is a very different scenario here.
        all_trainsets, testset = load_Femnist(
            total_agents,
            sizes=sizes,
            random_seed=seed,
            debug=debug,
        )
        return all_trainsets, testset
    elif dataset_name == "FemnistLabelSplit":
        # Since the sharding is done intresequely for optimization reasons, we load manually here
        # TODO: specify a manual train dir to mach to other number of nodes.
        all_trainsets, testset = load_Femnist_labelsplit(
            nb_nodes=total_agents,
            sizes=sizes,
            random_seed=seed,
            debug=debug,
        )
        return all_trainsets, testset
    if shards is None:
        raise ValueError(
            f"Got shards as None for {dataset_name} when only Femnist should be the case of None Shards."
        )
    trainset, testset = loader()
    c_len = len(trainset)
    if sizes is None:
        # Equal distribution of data among processes, from CIFAR10 and not a simplified function.
        # Speak with Rishi about having a separate function that performs this?
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


def deserialized_model(weights, model, shapes, lens):
    """
    Convert received dict to state_dict.

    Parameters
    ----------
    m : dict
        received dict

    Returns
    -------
    state_dict
        state_dict of received

    """
    state_dict = dict()
    start_index = 0
    for i, key in enumerate(model.state_dict()):
        end_index = start_index + lens[i]
        state_dict[key] = torch.from_numpy(
            weights[start_index:end_index].reshape(shapes[i])
        )
        start_index = end_index
    return state_dict


def load_model_from_path(model_path, model, shapes, lens, device=None):
    model_weigths = np.load(model_path)
    model.load_state_dict(deserialized_model(model_weigths, model, shapes, lens))
    if device is not None:
        model.to(device)


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
    experiment_path: str, machine_id: int, agent_id: int, attacks
) -> pd.DataFrame:
    """Generates dataframe for a given agent and machine id

    Args:
        experiment_path (str): The directory of the experiment
        machine_id (int): The id of the machine for the agent
        agent_id (int): The id of the agent on the machine
        attacks (str list): list of attacks to perform

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
    if not os.path.isdir(agent_models_directory):
        experiment_name = os.path.basename(experiment_path)
        for attack in attacks:
            assert os.path.exists(
                os.path.join(experiment_path, f"{attack}_{experiment_name}.csv")
            ), f"Models missing for {experiment_name} but the {attack} attack results are not there"
        return models_list
    for file in sorted(os.listdir(agent_models_directory)):
        file_attributes = get_model_attributes(file, agent_models_directory)
        models_list = pd.concat([models_list, file_attributes])
    return models_list


def get_all_models_properties(
    experiment_dir: str, nb_agents: int, nb_machines: int, attacks
) -> pd.DataFrame:
    """Generates a dataframe with all models path and properties (agent, iteration, target agent)

    Args:
        experiment_dir (str): The path to the experiment
        nb_agents (int): The total number of agent for the experiment
        nb_machines (int): The total number of machines for the experiment
        attacks (string list): List of attacks to perform

    Returns:
        pd.Dataframe: A dataframe containing a column for the model file,
            the iteration, the agent uid and the target agent uid.
    """
    models_df = pd.DataFrame({})
    for agent_uid in range(nb_agents):
        current_machine_id = agent_uid // (nb_agents // nb_machines)
        agent_id = agent_uid % (nb_agents // nb_machines)
        models_df = pd.concat(
            [
                models_df,
                list_models_path(experiment_dir, current_machine_id, agent_id, attacks),
            ]
        )

    return models_df


def get_all_experiments_properties(
    all_experiments_dir: str, nb_agents: int, nb_machines: int, attacks
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
            experiment_path,
            nb_agents=nb_agents,
            nb_machines=nb_machines,
            attacks=attacks,
        )
        current_experiment_df["experiment_name"] = experiment_name
        experiment_wide_df = pd.concat([experiment_wide_df, current_experiment_df])
    experiment_wide_df = experiment_wide_df.astype(
        {"iteration": "int32", "agent": "int32", "target": "int32"}
    )
    return experiment_wide_df


def main():
    DATASET = "Femnist"
    NB_CLASSES = POSSIBLE_DATASETS[DATASET][1]
    NB_AGENTS = 128
    NB_MACHINES = 8
    train_partitioner, test_data = load_dataset_partitioner(
        DATASET, NB_AGENTS, 90, 2, debug=True
    )
    nb_data = 0
    for agent in range(NB_AGENTS):
        train_data_current_agent = train_partitioner.use(agent)
        agent_classes_trainset = get_dataset_stats(train_data_current_agent, NB_CLASSES)
        print(f"Classes for agent {agent}: {agent_classes_trainset}")
        nb_data_agent = sum(agent_classes_trainset)
        print(f"Total number of data for agent {agent}: {nb_data_agent}")
        nb_data += nb_data_agent
    print(f"Total data for the {DATASET} dataset: {nb_data}")
    EXPERIMENT_DIR = "results/my_results/icml_experiments/cifar10/2067277_nonoise_dynamic_128nodes_1avgsteps_batch32_lr0.05_3rounds"

    all_models_df = get_all_models_properties(
        EXPERIMENT_DIR, NB_AGENTS, NB_MACHINES, ALL_ATTACKS
    )
    print(all_models_df)

    EXPERIMENTS_DIR = "results/my_results/test/testing_femnist_convergence_rates"

    t0 = time.time()
    all_experiments_df = get_all_experiments_properties(
        EXPERIMENTS_DIR, NB_AGENTS, NB_MACHINES, ALL_ATTACKS
    )
    t1 = time.time()
    print(all_experiments_df)
    print(f"All models paths retrieved in {t1-t0:.2f}s")


if __name__ == "__main__":
    main()
