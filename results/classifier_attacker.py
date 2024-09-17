import random
from typing import Optional

import load_experiments
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data


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
    NB: the authors lost access to the code.
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
    if debug:
        print(f"Generated losses of shape {all_losses_agent.shape}")
    return all_losses_agent


def split_data(dataset, nb_train, seed=421):
    generator = torch.Generator()
    generator.manual_seed(seed)
    shape = dataset.shape
    assert (
        shape[0] > nb_train
    ), f"Not enough training elements, expected {nb_train} but got shape {shape}"
    trainset, testset = torch.utils.data.random_split(
        dataset, [nb_train, shape[0] - nb_train], generator=generator
    )
    return trainset, testset


def main():
    # This is a debugging main, that loads a dummy dataset and launch an attack.
    # Full attack should be performed in "perform_attacks.py"
    import os

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

    model = model_type()
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    nb_params = sum([np.prod(p.size()) for p in model_params])
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

        trainset_for_train, trainset_for_test = split_data(
            losses_trainset_current_agent, nb_train=size_train
        )
        testset_for_train, testset_for_test = split_data(
            losses_testset_current_agent, nb_train=size_train
        )

        raise NotImplementedError
    return


if __name__ == "__main__":
    main()
