import os
import time

import load_experiments
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from decentralizepy.datasets.CIFAR10 import LeNet


def threshold_attack(local_train_losses, test_losses):
    # We need the losses to be increasing the more likely we are to be in the train set,
    # so 1-loss should be given as argument.
    num_true = local_train_losses.shape[0]
    # print("Number of training samples: ", num_true)

    assert (
        num_true <= test_losses.shape[0]
    ), f"Not enough test elements: {test_losses.shape[0]} when at least {num_true} where expected"
    y_true_balanced = torch.zeros((num_true + num_true,), dtype=torch.int32)
    y_true_balanced[:num_true] = 1

    y_pred_balanced = torch.zeros((num_true + num_true,), dtype=torch.float32)
    y_pred_balanced[:num_true] = local_train_losses
    y_pred_balanced[num_true:] = test_losses[
        torch.randperm(test_losses.shape[0])[:num_true]
    ]

    # Use the balanced y_pred for the ROC curve
    y_pred = y_pred_balanced.numpy()
    y_true = y_true_balanced.numpy()
    # print("Shapes: ", y_pred.shape, y_true.shape)

    # fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_true, y_pred)
    res = {
        # "fpr": fpr,
        # "tpr": tpr,
        # "thresholds": thresholds,
        "roc_auc": roc_auc,
    }
    # return res

    # Be carefull with this implementation:
    # if fpr and tpr needs to be returned, then this will break since the dimensions are different.
    return pd.DataFrame(res, index=[0])


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


def generate_shapes(model):
    shapes = []
    lens = []
    with torch.no_grad():
        for _, v in model.state_dict().items():
            shapes.append(v.shape)
            t = v.flatten().numpy()
            lens.append(t.shape[0])
    return shapes, lens


def load_model_from_path(model_path, model, shapes, lens, device=torch.device("cpu")):
    model_weigths = np.load(model_path)
    model.load_state_dict(deserialized_model(model_weigths, model, shapes, lens))
    model.to(device)


def generate_losses(
    model,
    dataset,
    loss_function=torch.nn.CrossEntropyLoss(reduction="none"),
    device=torch.device("cpu"),
):
    losses = torch.tensor([])
    with torch.no_grad():
        for x, y in dataset:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_function(y_pred, y)
            loss = loss.to("cpu")
            losses = torch.cat([losses, loss])
    return losses


def run_threshold_attack(
    running_model,
    model_path,
    trainset,
    testset,
    shapes,
    lens,
    device=torch.device("cpu"),
):
    load_model_from_path(
        model_path=model_path,
        model=running_model,
        shapes=shapes,
        lens=lens,
        device=device,
    )
    losses_train = generate_losses(running_model, trainset, device=device)
    losses_test = generate_losses(running_model, testset, device=device)
    # We use 1 - losses to have an AUC>0.5 (works for CrossEntropyLoss)
    results = threshold_attack(1 - losses_train, 1 - losses_test)
    return results


def attack_experiment(
    experiment_df,
    experiment_name: str,
    running_model,
    trainset_partitioner,
    testset,
    shapes,
    lens,
    batch_size,
    results_directory,
    device=torch.device("cpu"),
):
    print(f"Attacking {experiment_name}:  ", end="")
    res = pd.DataFrame({})
    current_experiment = experiment_df[
        experiment_df["experiment_name"] == experiment_name
    ]
    results_file = os.path.join(results_directory, experiment_name + ".csv")
    if os.path.exists(results_file):
        print(f"Attacks already computed for {experiment_name}, skipping")
        return

    print("Attack not already computed, starting.")
    for agent in sorted(pd.unique(current_experiment["agent"])):
        current_agent_experiments = current_experiment[
            current_experiment["agent"] == agent
        ]
        trainset = trainset_partitioner.use(agent)
        trainset = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=False
        )
        for _, row in current_agent_experiments.iterrows():
            iteration = row["iteration"]
            model_path = row["file"]
            target = row["target"]
            print(
                f"Launching attack {agent}->{target} at iteration {iteration}"
                + " " * 10,
                end="\r",
            )
            threshold_attack_result = run_threshold_attack(
                running_model,
                model_path,
                trainset,
                testset,
                shapes,
                lens,
                device=device,
            )
            threshold_attack_result["iteration"] = iteration
            threshold_attack_result["agent"] = agent
            threshold_attack_result["target"] = target
            res = pd.concat([res, threshold_attack_result])
    print(f"Writing results to {results_file}")
    res.to_csv(results_file)
    print(f"Finished attacking {experiment_name}")
    return


def main():
    BATCH_SIZE = 32
    DATASET = "CIFAR10"
    NB_CLASSES = load_experiments.POSSIBLE_DATASETS[DATASET][1]
    MODEL = load_experiments.POSSIBLE_DATASETS[DATASET][2]
    NB_AGENTS = 128
    NB_MACHINES = 8
    SEED = 90
    KSHARDS = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DEVICE = torch.device("cpu")
    print(f"Running on {DEVICE}")

    MODEL_PATH = "results/my_results/icml_experiments/cifar10/2069897_muffliato_dynamic_128nodes_10avgsteps_16th/machine0/attacked_model/machine0/0/model_it10000_0_to36.npy"
    CURRENT_AGENT = 0
    train_partitioner, testset = load_experiments.load_dataset_partitioner(
        DATASET, NB_AGENTS, SEED, KSHARDS
    )
    MODEL = LeNet()
    SHAPES, LENS = generate_shapes(MODEL)

    trainset = train_partitioner.use(CURRENT_AGENT)
    stats_before = load_experiments.get_dataset_stats(trainset, NB_CLASSES)
    print(f"Dataset stats for user {CURRENT_AGENT} before batch: {stats_before}")
    trainset = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=False
    )

    stats = load_experiments.get_dataset_stats_batch(trainset, NB_CLASSES)
    print(f"Dataset stats for user {CURRENT_AGENT} after batch: {stats}")
    testset = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    t0 = time.time()
    threshold_attack_result = run_threshold_attack(
        MODEL, MODEL_PATH, trainset, testset, SHAPES, LENS, device=DEVICE
    )
    t1 = time.time()

    print(threshold_attack_result["roc_auc"])
    print(f"Attack done in {t1-t0:.2f}s")

    # ---------
    # Running the main attacks on all the experiments
    # ---------
    print("---- Starting main attacks ----")
    EXPERIMENTS_DIR = "results/my_results/icml_experiments/cifar10/"
    RESULTS_PATH = "results/my_results/icml_experiments/cifar10_attack_results/"
    print("Loading experiments dirs")
    all_experiments_df = load_experiments.get_all_experiments_properties(
        EXPERIMENTS_DIR, NB_AGENTS, NB_MACHINES
    )

    all_experiments_df.reset_index()

    # RESULTS_PATH = f"results/my_results/icml_experiments/cifar10_attack_results_quick/"
    # all_experiments_df = all_experiments_df[
    #     all_experiments_df["iteration"].isin([500, 5000, 10000])
    # ]

    times = []
    print("Loaded experiments setup, launching attacks")
    for experiment_name in sorted(pd.unique(all_experiments_df["experiment_name"])):
        t0 = time.time()
        attack_experiment(
            all_experiments_df,
            experiment_name=experiment_name,
            running_model=MODEL,
            trainset_partitioner=train_partitioner,
            testset=testset,
            shapes=SHAPES,
            lens=LENS,
            batch_size=BATCH_SIZE,
            results_directory=RESULTS_PATH,
            device=DEVICE,
        )
        t1 = time.time()
        times.append(t1 - t0)
        print(f"Took {times[-1]/60:.2f} minutes")

    print(f"Total time: {sum(times)/(60*60):.2f} hours")
    print(
        f"Average time: {sum(times)/len(sorted(pd.unique(all_experiments_df['experiment_name'])))* 1/60:.2f} minutes"
    )

    # import matplotlib.pyplot as plt

    # plt.figure()
    # plt.plot(
    #     threshold_attack_result["fpr"],
    #     threshold_attack_result["tpr"],
    #     color="darkorange",
    #     lw=2,
    #     label="ROC curve (AUC = %0.2f)" % threshold_attack_result["roc_auc"],
    # )
    # plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    # # plt.xscale('log')
    # # plt.yscale('log')
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver Operating Characteristic")
    # plt.legend(loc="lower right")
    # plt.show()


if __name__ == "__main__":
    main()
