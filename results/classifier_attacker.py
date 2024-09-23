import random
import time
from typing import Optional

import load_experiments
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader, Dataset


class ConcatWithLabels(Dataset):
    """Data class that will consider concatenated datasets, and add as post processing.

    Args:
        torch (_type_): _description_
    """

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        x = self.x_train[idx]
        y = self.y_train[idx]
        return x, y


class SimpleAttacker(nn.Module):
    def __init__(self, nb_in):
        super().__init__()
        self.fc1 = nn.Linear(nb_in, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze(1)
        # Removed as I use CrossEntropyLoss
        x = self.softmax(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=8, padding=4)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class FCNAttacker(nn.Module):
    """Classifier used "Efficient passive membership inference attack in federated learning".
    See https://arxiv.org/abs/2111.00430.
    Parameters (kernel size and other) obtained after discussion with the author.
    NB: the authors seems to have lost access to the code.
    """

    def __init__(self):
        # 1,316,866 parameters.
        super(FCNAttacker, self).__init__()
        self.block1 = ConvBlock(1, 128)
        self.block2 = ConvBlock(128, 256)
        self.block3 = ConvBlock(256, 128)
        self.bn_final = nn.BatchNorm1d(128)
        self.fc = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.bn_final(x)
        x = x.mean(dim=2)  # Global average pooling
        x = self.fc(x)
        x = self.softmax(x)
        return x


def train_classifier(
    model: nn.Module,
    dataset: DataLoader,
    nb_epochs: int,
    device: torch.device,
    lr: float = 0.001,
    momentum: float = 0.9,
):
    model.to(device)
    loss_function = nn.CrossEntropyLoss()  # This may be a
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    train_losses = []

    for epoch in range(nb_epochs):
        print(f"Training iteration {epoch}.")
        model.train()

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataset):
            inputs, labels = inputs.to(device), labels.to(device)
            # Add channel dimension, now shape becomes [batch_size, 1, sequence_length]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss)
    print(f"Finished training. Running loss on last iteration: {running_loss}")
    return train_losses


def eval_classifier(model: nn.Module, testloader: DataLoader, device: torch.device):
    model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    total_correct = 0
    total_predicted = 0

    # Initialize counters for precision and recall computation
    total_pred = [0, 0]  # Total number of samples for each class
    correct_pred = [0, 0]  # True positives for each class
    false_positives = [0, 0]  # False positives for each class
    false_negatives = [0, 0]  # False negatives for each class

    with torch.no_grad():
        loss_val = 0.0
        count = 0
        for elems, labels in testloader:
            elems, labels = elems.to(device), labels.to(device)
            outputs = model(elems)
            loss = loss_function(outputs, labels)
            loss_val += loss.item()
            count += 1

            # Get the predicted classes
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                # Check if prediction is correct
                if label == prediction:
                    correct_pred[label] += 1
                    total_correct += 1
                else:
                    false_positives[prediction] += 1
                    false_negatives[label] += 1
                total_pred[label] += 1
                total_predicted += 1

    accuracy = total_correct / total_predicted * 100

    # Calculate precision and recall for each class
    precision = [0.0, 0.0]
    recall = [0.0, 0.0]

    for i in range(2):  # Assuming binary classification (two classes: 0 and 1)
        if (correct_pred[i] + false_positives[i]) > 0:
            precision[i] = correct_pred[i] / (correct_pred[i] + false_positives[i])
        if (correct_pred[i] + false_negatives[i]) > 0:
            recall[i] = correct_pred[i] / (correct_pred[i] + false_negatives[i])

    print(f"Test loss: {loss_val}; test accuracy: {accuracy:.2f}%")
    print(
        f"Precision for class 0: {precision[0]*100:.2f}%, class 1: {precision[1]*100:.2f}%"
    )
    print(f"Recall for class 0: {recall[0]*100:.2f}%, class 1: {recall[1]*100:.2f}%")

    return loss_val, accuracy, precision, recall, correct_pred, total_pred


def update_data(
    current_data: Optional[torch.Tensor],
    model: torch.nn.Module,
    dataset,
    loss_function,
    device=torch.device("cpu"),
    debug=False,
):
    losses, _, _ = load_experiments.generate_losses(
        model, dataset, loss_function=loss_function, device=device, debug=debug
    )

    # Remove nans
    losses, _ = load_experiments.filter_nans(
        losses,
        classes=None,
        debug_name="update_data",
        loss_type=type(loss_function),
    )
    if len(losses) == 0:
        return current_data

    losses = losses.unsqueeze(1)
    if current_data is None:
        return losses
    current_data = torch.cat((current_data, losses), dim=1)
    return current_data


def generate_time_series_dataset(
    dataset,
    agent_model_properties: pd.DataFrame,
    attacked_model,
    loss_function,
    shapes,
    lens,
    device,
    debug=False,
):

    all_losses_agent = None
    # Sort so that we have a time series.
    agent_model_properties.sort_values("iteration", inplace=True)
    for _, line in agent_model_properties.iterrows():
        # Used to ensure everything is in the correct order.
        iteration = line["iteration"]

        # Load the model of node agent at iteration iteration
        # TODO: loading time could be reduced by doing both trainset and testset losses at the same time,
        # do it if such a level of optimisation is necessary.
        load_experiments.load_model_from_path(
            line["file"], attacked_model, shapes=shapes, lens=lens, device=device
        )

        all_losses_agent = update_data(
            all_losses_agent,
            model=attacked_model,
            dataset=dataset,
            loss_function=loss_function,
            device=device,
            debug=debug,
        )
    assert all_losses_agent is not None
    # Add a channel dimension that will be needed by the network. We only have 1 channel (the loss)
    all_losses_agent = all_losses_agent.unsqueeze(1)
    if debug:
        print(f"Generated losses of shape {all_losses_agent.shape}")
    return all_losses_agent


def split_data(dataset, nb_train, generator):
    shape = dataset.shape
    assert (
        shape[0] > nb_train
    ), f"Not enough training elements, expected {nb_train} but got shape {shape}"
    splits = torch.utils.data.random_split(
        dataset, [nb_train, shape[0] - nb_train], generator=generator
    )
    trainset, testset = splits[0], splits[1]
    return trainset, testset


def generate_attacker_dataset(
    losses_trainset, losses_testset, size_train, batch_size=256, seed=421
):
    generator = torch.Generator()
    generator.manual_seed(seed)
    trainset_for_train, trainset_for_test = split_data(
        losses_trainset, nb_train=size_train, generator=generator
    )
    testset_for_train, testset_for_test = split_data(
        losses_testset, nb_train=size_train, generator=generator
    )

    x_train = torch.utils.data.ConcatDataset((trainset_for_train, testset_for_train))
    y_train = torch.cat(
        [
            torch.zeros(
                len(trainset_for_train),
                dtype=torch.long,  # Necessary for crossentropyloss
            ),
            torch.ones(
                len(testset_for_train),
                dtype=torch.long,  # Necessary for crossentropyloss
            ),
        ]
    )
    dataloader_train = DataLoader(
        ConcatWithLabels(x_train, y_train),
        shuffle=True,
        num_workers=0,
        batch_size=batch_size,
    )

    x_test = torch.utils.data.ConcatDataset((trainset_for_test, testset_for_test))
    y_test = torch.cat(
        [
            torch.zeros(
                len(trainset_for_test),
                dtype=torch.long,  # Necessary for crossentropyloss
            ),
            torch.ones(
                len(testset_for_test),
                dtype=torch.long,  # Necessary for crossentropyloss
            ),
        ]
    )
    dataloader_test = DataLoader(
        (ConcatWithLabels(x_test, y_test)),
        shuffle=True,
        num_workers=0,
        batch_size=batch_size,
    )

    return dataloader_train, dataloader_test


def main(dataset_name="CIFAR10"):
    # This is a debugging main, that loads a dummy dataset and launch an attack.
    # Full attack should be performed in "perform_attacks.py"
    import os

    import matplotlib.pyplot as plt

    debug = True
    model_type = FCNAttacker

    device = torch.device("cuda")
    batch_size = 4096
    loss_function = torch.nn.CrossEntropyLoss(reduction="none")
    size_train = 2000  # 2000 in the original paper
    nb_epoch = 300
    lr = 0.005
    momentum = 0.8

    # CIFAR test experiment
    if dataset_name == "CIFAR10":
        nb_agents = 128
        nb_machines = 4
        seed = 90
        kshards = 2
        size_train = 200  # Downscaled for testing purposes: impossible to do for CIFAR split accross 128 nodes.
        batch_size = 16
        experiment_dir = "results/my_results/test/fixing_attacks/cifar/4849292_cifar_nonoise_128nodes_1avgsteps_static_seed90_degree6_LeNet_lr0.05_3rounds"
        attacked_model = load_experiments.LeNet()

    # FemnistLabelSplit test experiment.
    elif dataset_name == "FemnistLabelSplit":
        nb_agents = 64
        nb_machines = 2
        seed = 90
        kshards = None
        # experiment_dir = "results/my_results/test/fixing_attacks/femnist_labelsplit/4860405_femnistLabelSplit_nonoise_64nodes_1avgsteps_static_seed90_degree4_RNET_lr0.01_3rounds"
        experiment_dir = "results/my_results/test/fixing_attacks/femnist_labelsplit/4862162_femnistLabelSplit_zerosum_selfnoise_64nodes_1avgsteps_16th_static_seed90_degree4_RNET_lr0.01_3rounds"
        attacked_model = load_experiments.RNET()

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    experiment_name = os.path.basename(experiment_dir)

    model = model_type()
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    nb_params = sum([np.prod(p.size()) for p in model_params])
    print(model)
    print(f"{nb_params:,d} trainable parameters.")

    shapes, lens = load_experiments.generate_shapes(attacked_model)

    trainset_partitioner, testset = load_experiments.load_dataset_partitioner(
        dataset_name, nb_agents, seed, kshards, debug=debug
    )

    testset = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    assert os.path.exists(experiment_dir)
    models_properties = load_experiments.get_all_models_properties(
        experiment_dir=experiment_dir,
        nb_agents=nb_agents,
        nb_machines=nb_machines,
        attacks=["classifier"],
    )
    print(models_properties)

    groups = models_properties.groupby("agent")
    for agent, agent_model_properties in groups:
        nb_models = len(agent_model_properties)
        trainset_current_agent = trainset_partitioner.use(agent)
        trainset_current_agent = torch.utils.data.DataLoader(
            trainset_current_agent, batch_size=batch_size, shuffle=False
        )

        if debug:
            print(f"Generating train losses for {agent}")
        losses_trainset_current_agent = generate_time_series_dataset(
            trainset_current_agent,
            agent_model_properties,
            attacked_model,
            loss_function,
            shapes,
            lens,
            device,
            debug=debug,
        )

        if debug:
            print(f"Generating test losses for {agent}")
        losses_testset_current_agent = generate_time_series_dataset(
            testset,
            agent_model_properties,
            attacked_model,
            loss_function,
            shapes,
            lens,
            device,
            debug=debug,
        )

        attacker_trainset, attacker_testset = generate_attacker_dataset(
            losses_trainset_current_agent,
            losses_testset_current_agent,
            size_train=size_train,
        )
        # Reset the model to start from fresh parameters
        model = model_type()  # TODO: change this

        print(f"Launching training {agent}")
        t0 = time.time()
        train_losses = train_classifier(
            model,
            attacker_trainset,
            device=device,
            nb_epochs=nb_epoch,
            lr=lr,
            momentum=momentum,
        )
        t1 = time.time()
        print(f"Training took {(t1-t0)/60:.2f}min")

        res = eval_classifier(model, attacker_testset, device=device)
        plt.plot(train_losses)
        plt.title(f"{experiment_name} - Train loss evolution")
        plt.show()
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        choices=["CIFAR10", "FemnistLabelSplit"],
    )

    args = parser.parse_args()

    dataset = args.dataset

    main(dataset)
