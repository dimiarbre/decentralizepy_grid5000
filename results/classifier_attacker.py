import random
from typing import Optional

import load_experiments
import numpy as np
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
    # model.eval()
    total_correct = 0
    total_predicted = 0
    total_pred = [0, 0]
    correct_pred = [0, 0]
    with torch.no_grad():
        loss_val = 0.0
        count = 0
        for elems, labels in testloader:
            elems, labels = elems.to(device), labels.to(device)
            outputs = model(elems)
            loss = loss_function(outputs, labels)
            loss_val += loss.item()
            count += 1
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                # logging.debug("{} predicted as {}".format(label, prediction))
                if label == prediction:
                    correct_pred[label] += 1
                    total_correct += 1
                total_pred[label] += 1
                total_predicted += 1
    accuracy = total_correct / total_predicted * 100
    print(
        f"Test loss: {loss_val}; accuracy: {accuracy:.2f}%; predictions: {total_pred}"
    )
    return loss_val, accuracy, correct_pred, total_pred


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
    losses = losses.unsqueeze(1)
    if current_data is None:
        return losses
    current_data = torch.cat((current_data, losses), dim=1)
    return current_data


def generate_time_series_dataset(
    dataset,
    agent_model_properties,
    attacked_model,
    loss_function,
    shapes,
    lens,
    device,
    debug=False,
):

    all_losses_agent = None
    for _, line in agent_model_properties.iterrows():
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
    y_train = torch.zeros(
        len(x_train),
        dtype=torch.long,  # Necessary for crossentropyloss
    )
    dataloader_train = DataLoader(
        ConcatWithLabels(x_train, y_train),
        shuffle=True,
        num_workers=0,
        batch_size=batch_size,
    )

    x_test = torch.utils.data.ConcatDataset((trainset_for_test, testset_for_test))
    y_test = torch.ones(
        len(x_test),
        dtype=torch.long,  # Necessary for crossentropyloss
    )
    dataloader_test = DataLoader(
        (ConcatWithLabels(x_test, y_test)),
        shuffle=True,
        num_workers=0,
        batch_size=batch_size,
    )

    return dataloader_train, dataloader_test


def main():
    # This is a debugging main, that loads a dummy dataset and launch an attack.
    # Full attack should be performed in "perform_attacks.py"
    import os

    import matplotlib.pyplot as plt

    nb_agents = 128
    nb_machines = 4
    seed = 90
    kshards = 2
    debug = True
    experiment_dir = "results/my_results/test/fixing_attacks/cifar/4849292_cifar_nonoise_128nodes_1avgsteps_static_seed90_degree6_LeNet_lr0.05_3rounds"
    model_type = FCNAttacker
    attacked_model = load_experiments.LeNet()
    device = torch.device("cuda")
    batch_size = 128
    loss_function = torch.nn.CrossEntropyLoss(reduction="none")
    size_train = 200  # 2000 in the original paper, downscaled for testing purposes: impossible to do for CIFAR split accross 128 nodes.
    nb_epoch = 100

    # TODO: remove this, as it should be computed or passed through directly.
    model = model_type()
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    nb_params = sum([np.prod(p.size()) for p in model_params])
    print(model)
    print(f"{nb_params:,d} trainable parameters.")

    shapes, lens = load_experiments.generate_shapes(attacked_model)

    trainset_partitioner, testset = load_experiments.load_dataset_partitioner(
        "CIFAR10", nb_agents, seed, kshards, debug=debug
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
        train_losses = train_classifier(
            model, attacker_trainset, device=device, nb_epochs=nb_epoch
        )

        res = eval_classifier(model, attacker_testset, device=device)
        plt.plot(train_losses)
        plt.legend("Train loss evolution")
        plt.show()
        raise NotImplementedError
    return


if __name__ == "__main__":
    main()
