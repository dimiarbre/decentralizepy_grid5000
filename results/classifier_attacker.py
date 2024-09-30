import os
import random
import time
from math import floor
from typing import Optional

import load_experiments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ConcatWithLabels(Dataset):
    """Data class that will consider concatenated datasets, and add as post processing.

    Args:
        torch (_type_): _description_
    """

    def __init__(
        self,
        x_train: torch.utils.data.ConcatDataset,
        y_train: torch.Tensor,
    ):
        self.x_train = x_train
        self.y_train = y_train

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        x = self.x_train[idx]
        y = self.y_train[idx]
        return x, y


def fast_collate(batch):
    """
    Optimized collate function to concatenate tensor batches.
    Assuming all samples in `batch` are already torch tensors.
    """
    x_batch = torch.stack([item[0] for item in batch])  # Stack all data samples
    y_batch = torch.stack([item[1] for item in batch])  # Stack all label samples
    return x_batch, y_batch


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
        # Removed since we use CrossEntropyLoss
        # x = self.softmax(x)
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

    def __init__(self, nb_in=None):
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

        # Removed since we decided to use CrossEntropyLoss.
        # The original author's model appeared to have this layer.
        # x = self.softmax(x)
        return x


def train_classifier(
    model: nn.Module,
    dataset: DataLoader,
    nb_epochs: int,
    device: torch.device,
    lr: float = 0.001,
    momentum: float = 0.9,
    weight: Optional[torch.Tensor] = None,
    debug=False,
):
    """Train a classifier

    Args:
        model (nn.Module): the initailized model
        dataset (DataLoader): The trainset
        nb_epochs (int): number of training iterations
        device (torch.device): Device to do the training on.
        lr (float, optional): Learning rate. Defaults to 0.001.
        momentum (float, optional): Momentum. Defaults to 0.9.
        weight (Optional[torch.Tensor], optional): Counter weights to the training class to balance classes. Defaults to None.
        debug (bool, optional): Extra debugging prints. Defaults to False.

    Returns:
        train_losses (List[float]): list of all train losses.
    """
    model.to(device)
    if weight is not None:
        weight.to(device)
        if debug:
            print(f"Reweighting losses: {weight}")
    loss_function = nn.CrossEntropyLoss(weight=weight)
    loss_function.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    train_losses = []
    for epoch in tqdm(range(nb_epochs)):
        # print(f"Training iteration {epoch}.")
        model.train()

        running_loss = 0.0
        num_batches = 0
        for i, (inputs, labels) in enumerate(dataset):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1
        # Normalize the loss by the number of batches
        average_loss = running_loss / num_batches if num_batches > 0 else 0
        train_losses.append(average_loss)
    print(f"Finished training. Running loss on last iteration: {average_loss}")
    return train_losses


def eval_classifier(model: nn.Module, testloader: DataLoader, device: torch.device):
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    total_correct = 0
    total_predicted = 0

    # Initialize counters for precision and recall computation
    total_pred = [0, 0]  # Total number of samples for each class
    correct_pred = [0, 0]  # True positives for each class
    false_positives = [0, 0]  # False positives for each class
    false_negatives = [0, 0]  # False negatives for each class

    y_true = torch.tensor([], dtype=torch.long, device=device)
    y_pred = torch.tensor([], dtype=torch.long, device=device)
    y_proba = torch.tensor([], dtype=torch.float32, device=device)

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

            # TODO: may need to switch to class 0. Depends on "generate_y_data"
            probabilities = F.softmax(outputs, dim=1)[:, 1]

            y_true = torch.cat([y_true, labels])
            y_pred = torch.cat([y_pred, predictions])
            y_proba = torch.cat([y_proba, probabilities])

            for label, prediction in zip(labels, predictions):

                # Check if prediction is correct
                if label == prediction:
                    correct_pred[label] += 1
                    total_correct += 1
                else:
                    false_positives[prediction] += 1
                    false_negatives[label] += 1
                total_pred[prediction] += 1
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

    print(
        f"Test loss: {loss_val/count}; test accuracy: {accuracy:.2f}%. Predictions: {total_pred}"
    )
    print(
        f"Precision for class 0: {precision[0]*100:.2f}%, class 1: {precision[1]*100:.2f}%"
    )
    print(f"Recall for class 0: {recall[0]*100:.2f}%, class 1: {recall[1]*100:.2f}%")

    conf_matrix = confusion_matrix(y_true.cpu(), y_pred.cpu(), normalize="true")

    return (
        loss_val / count,
        accuracy,
        precision,
        recall,
        correct_pred,
        total_pred,
        conf_matrix,
        (y_true, y_proba),
    )


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
    shapes_target,
    lens_target,
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
            line["file"],
            attacked_model,
            shapes=shapes_target,
            lens=lens_target,
            device=device,
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


def generate_y_data(trainset, testset):
    y_trainset = torch.ones(
        len(trainset),
        dtype=torch.long,  # Necessary for crossentropyloss
    )
    y_testset = torch.zeros(
        len(testset),
        dtype=torch.long,  # Necessary for crossentropyloss
    )

    y = torch.cat([y_trainset, y_testset], dim=0)
    trainset_size = len(trainset)
    testset_size = len(testset)
    total_size = trainset_size + testset_size
    weight = torch.tensor(  # Ensure this repartition of classes matches the one of y_trainset and y_testset
        [
            total_size / testset_size,
            total_size / trainset_size,
        ],
        dtype=torch.float32,
    )
    return y, weight


def generate_attacker_dataset(
    losses_trainset, losses_testset, fractions, batch_size=256, seed=421
):
    generator = torch.Generator()
    generator.manual_seed(seed)
    nb_training_samples_from_train = floor(len(losses_trainset) * fractions)
    trainset_for_train, trainset_for_test = split_data(
        losses_trainset, nb_train=nb_training_samples_from_train, generator=generator
    )
    nb_training_samples_from_test = floor(len(losses_testset) * fractions)
    testset_for_train, testset_for_test = split_data(
        losses_testset, nb_train=nb_training_samples_from_test, generator=generator
    )

    total_attacker_trainset_size = (
        nb_training_samples_from_train + nb_training_samples_from_test
    )

    x_train = torch.utils.data.ConcatDataset((trainset_for_train, testset_for_train))
    y_train, weight = generate_y_data(
        trainset=trainset_for_train, testset=testset_for_train
    )

    # TODO: set a random seed to be able to reproduce results?
    dataloader_train = DataLoader(
        ConcatWithLabels(x_train, y_train),
        shuffle=True,
        num_workers=0,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=fast_collate,
    )

    x_test = torch.utils.data.ConcatDataset((trainset_for_test, testset_for_test))
    y_test, _ = generate_y_data(trainset=trainset_for_test, testset=testset_for_test)
    dataloader_test = DataLoader(
        (ConcatWithLabels(x_test, y_test)),
        shuffle=True,
        num_workers=0,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=fast_collate,
    )

    return dataloader_train, dataloader_test, weight


def classifier_attack(
    agent_model_properties,
    trainset,
    testset,
    loss_function,
    attacked_model,
    shapes_target,
    lens_target,
    device,
    agent,
    attacker_model_initializer,
    fractions: float,
    nb_epoch: int,
    batch_size=256,
    lr=0.01,
    momentum=0.9,
    debug=False,
):
    if debug:
        print(f"Generating train losses for {agent}")
    losses_trainset_current_agent = generate_time_series_dataset(
        trainset,
        agent_model_properties,
        attacked_model,
        loss_function,
        shapes_target,
        lens_target,
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
        shapes_target,
        lens_target,
        device,
        debug=debug,
    )

    attacker_trainset, attacker_testset, weight = generate_attacker_dataset(
        losses_trainset_current_agent,
        losses_testset_current_agent,
        fractions=fractions,
        batch_size=batch_size,
        seed=421,
    )

    # This handles cases where there are Nans or invalid models.
    assert (
        losses_trainset_current_agent.shape[1:]
        == losses_testset_current_agent.shape[1:]
    )
    nb_dimensions = losses_trainset_current_agent.shape[2]
    print(f"in_dimension: {nb_dimensions}")

    # Reset the model to start from fresh parameters
    attacker_model = attacker_model_initializer(nb_in=nb_dimensions)

    model_params = filter(lambda p: p.requires_grad, attacker_model.parameters())
    if debug:
        nb_params = sum([np.prod(p.size()) for p in model_params])
        print(attacker_model)
        print(f"{nb_params:,d} trainable parameters.")
        print(f"Launching training {agent}")

    t0 = time.time()
    train_losses = train_classifier(
        attacker_model,
        attacker_trainset,
        device=device,
        nb_epochs=nb_epoch,
        lr=lr,
        momentum=momentum,
        weight=weight,
        debug=debug,
    )
    if debug:
        t1 = time.time()
        print(f"Training took {(t1-t0)/60:.2f}min")
    res = eval_classifier(attacker_model, attacker_testset, device=device)
    return res, train_losses


def generate_statistics_figure(
    experiment_name,
    experiment_dir,
    train_losses,
    conf_matrix,
    y_true,
    y_proba,
    roc_auc,
    accuracy,
    node_id,
):
    fig, axs = plt.subplots(2, 2)

    # Plot the evolution of the train loss
    ax = axs[0, 0]
    ax.plot(train_losses)
    ax.set_title(f"Train loss evolution. Final accuracy: {accuracy:.2f}%")

    # Plot the confusion matrix
    ax = axs[1, 0]
    ConfusionMatrixDisplay(confusion_matrix=conf_matrix).plot(ax=ax, cmap="Blues")
    ax.set_title("Normalized confusion matrix")

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_proba)
    # Plot the ROC-AUC
    ax = axs[0, 1]
    ax.plot(fpr, tpr, "b")

    # current_axs.legend(loc="lower right")
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    ax.set_title(f"ROC-Curve - AUC: {roc_auc}")

    # Plot the log ROC-AUC
    ax = axs[1, 1]

    ax.plot(fpr, tpr, "b")

    # current_axs.legend(loc="lower right")
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim([1e-5, 1])
    ax.set_ylim([1e-5, 1])
    ax.set_title("Log scale ROC-Curve")

    # Adjust the layout if necessary
    fig.suptitle(f"{experiment_name}")
    plt.tight_layout()

    figpath = os.path.join(
        experiment_dir, f"classifier_attack_summary_node{node_id}.png"
    )
    fig.savefig(figpath)

    plt.close(fig)
    return


def sample_specific_scores(
    tpr: np.ndarray, fpr: np.ndarray, fpr_to_consider: list[float]
):
    """
    Returns the true positive rates (tpr) corresponding to specific
    false positive rates (fpr_to_consider), by finding the closest values in the given fpr list.

    Args:
        tpr (np.ndarray): True positive rates.
        fpr (np.ndarray): False positive rates (same length as tpr).
        fpr_to_consider (list of float): False positive rates at which to sample the tpr.

    Returns:
        dictionnary of floats: The tpr values corresponding to the closest fpr values.
    """
    sampled_tpr = {}

    for target_fpr in fpr_to_consider:
        if target_fpr < fpr[1]:
            # fpr[0] is always 0.
            # We remove data point corresponding to no data point.
            # TODO: ensure I really want to keep this code fragment.
            sampled_tpr[target_fpr] = torch.nan
            continue

        # Find the index of the closest fpr value in the original fpr list
        closest_idx = np.argmin(np.abs(fpr - target_fpr))

        sampled_tpr[target_fpr] = tpr[closest_idx]

    return sampled_tpr


def run_classifier_attack(
    models_properties: pd.DataFrame,
    experiment_dir,
    trainset_partitioner,
    testset,
    loss_function,
    attacked_model,
    shapes_target,
    lens_target,
    device,
    batch_size,
    attacker_model_initializer=SimpleAttacker,
    fractions=0.7,
    nb_epoch=300,
    lr=0.01,
    momentum=0.9,
    debug=False,
):
    experiment_name = os.path.basename(experiment_dir)
    groups = models_properties.groupby("agent")
    all_results = pd.DataFrame({})
    for agent, agent_model_properties in groups:
        targets = agent_model_properties["target"].to_numpy()
        # TODO: this probably won't work on a dynamic topology, I'll probably need to do a second group by
        # Or revamp how models are logged (which is already a todo)
        assert (targets[0] == targets).all(), f"Not uniform targets {targets}"
        target = targets[0]
        trainset_local_node = trainset_partitioner.use(agent)
        trainset_local_node = DataLoader(
            trainset_local_node, batch_size=batch_size, shuffle=False
        )
        current_res, train_losses = classifier_attack(
            agent_model_properties,
            trainset=trainset_local_node,
            testset=testset,
            loss_function=loss_function,
            attacked_model=attacked_model,
            shapes_target=shapes_target,
            lens_target=lens_target,
            device=device,
            agent=agent,
            attacker_model_initializer=attacker_model_initializer,
            fractions=fractions,
            nb_epoch=nb_epoch,
            batch_size=batch_size,
            lr=lr,
            momentum=momentum,
            debug=debug,
        )
        (
            loss_val,
            accuracy,
            precision,
            recall,
            correct_pred,
            total_pred,
            conf_matrix,
            (y_true, y_proba),
        ) = current_res

        y_true, y_proba = y_true.cpu(), y_proba.cpu()
        roc_auc = roc_auc_score(y_true=y_true, y_score=y_proba)
        fpr, tpr, _ = roc_curve(y_true, y_proba)

        # TODO: make this a global variable?
        fpr_to_consider = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

        scores = sample_specific_scores(
            tpr=tpr, fpr=fpr, fpr_to_consider=fpr_to_consider
        )

        # Generate visualization plots.
        generate_statistics_figure(
            experiment_name=experiment_name,
            experiment_dir=experiment_dir,
            train_losses=train_losses,
            conf_matrix=conf_matrix,
            y_true=y_true,
            y_proba=y_proba,
            roc_auc=roc_auc,
            accuracy=accuracy,
            node_id=agent,
        )

        # Organize results in the DF.
        current_res = {}
        current_res["agent"] = [agent]
        current_res["roc_auc"] = [roc_auc]
        current_res["target"] = [target]
        current_res["attacker_model"] = [attacker_model_initializer.__name__]

        for fpr_target, tpr_value in scores.items():
            current_res[f"tpr_at_fpr{fpr_target}"] = tpr_value

        df_current_res = pd.DataFrame(current_res)
        df_current_res.set_index("agent")
        all_results = pd.concat([all_results, df_current_res])

    return all_results


def main(dataset_name="CIFAR10"):
    # This is a debugging main, that loads a dummy dataset and launch an attack.
    # Full attack should be performed in "perform_attacks.py"

    debug = True
    model_type = SimpleAttacker
    # model_type = FCNAttacker

    device = torch.device("cuda")
    batch_size = 4096
    loss_function = torch.nn.CrossEntropyLoss(reduction="none")
    size_train = 2000  # 2000 in the original paper
    fractions = 0.7
    nb_epoch = 300
    lr = 0.01
    momentum = 0.9

    # CIFAR test experiment
    if dataset_name == "CIFAR10":
        nb_agents = 128
        nb_machines = 4
        seed = 90
        kshards = 2
        batch_size = 512
        experiment_dir = "results/my_results/test/fixing_attacks/cifar/4849292_cifar_nonoise_128nodes_1avgsteps_static_seed90_degree6_LeNet_lr0.05_3rounds"
        attacked_model = load_experiments.LeNet()

    # FemnistLabelSplit test experiment.
    elif dataset_name == "FemnistLabelSplit":
        nb_agents = 64
        nb_machines = 2
        seed = 90
        kshards = None
        # experiment_dir = "results/my_results/test/fixing_attacks/femnist_labelsplit/4860405_femnistLabelSplit_nonoise_64nodes_1avgsteps_static_seed90_degree4_RNET_lr0.01_3rounds"
        # experiment_dir = "results/my_results/test/fixing_attacks/femnist_labelsplit/4862162_femnistLabelSplit_zerosum_selfnoise_64nodes_1avgsteps_16th_static_seed90_degree4_RNET_lr0.01_3rounds"
        experiment_dir = "results/my_results/test/fixing_attacks/femnist_labelsplit/4861560_femnistLabelSplit_muffliato_64nodes_5avgsteps_16th_static_seed90_degree4_RNET_lr0.01_3rounds"
        attacked_model = load_experiments.RNET()

    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    experiment_name = os.path.basename(experiment_dir)

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

    res = run_classifier_attack(
        models_properties=models_properties,
        experiment_dir=experiment_dir,
        trainset_partitioner=trainset_partitioner,
        testset=testset,
        loss_function=loss_function,
        attacked_model=attacked_model,
        shapes_target=shapes,
        lens_target=lens,
        device=device,
        batch_size=batch_size,
        attacker_model_initializer=model_type,
        fractions=fractions,
        nb_epoch=nb_epoch,
        lr=lr,
        momentum=momentum,
        debug=debug,
    )
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        choices=["CIFAR10", "FemnistLabelSplit", "Femnist"],
    )

    args = parser.parse_args()

    dataset = args.dataset

    main(dataset)
