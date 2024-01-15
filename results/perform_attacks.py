import concurrent.futures
import copy
import multiprocessing
import os

import load_experiments
import numpy as np
import pandas as pd
import torch
from LinkabilityAttack import LinkabilityAttack
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from decentralizepy.datasets.CIFAR10 import LeNet


def threshold_attack(local_train_losses, test_losses, balanced=True):
    # We need the losses to be increasing the more likely we are to be in the train set,
    # so 1-loss should be given as argument.
    num_true = local_train_losses.shape[0]
    # print("Number of training samples: ", num_true)

    assert (
        num_true <= test_losses.shape[0]
    ), f"Not enough test elements: {test_losses.shape[0]} when at least {num_true} where expected"

    if balanced:
        y_true = torch.zeros((num_true + num_true,), dtype=torch.int32)
        y_pred = torch.zeros((num_true + num_true,), dtype=torch.float32)
    else:
        y_true = torch.zeros((num_true + test_losses.shape[0]), dtype=torch.int32)
        y_pred = torch.zeros((num_true + test_losses.shape[0]), dtype=torch.float32)

    y_true[:num_true] = 1
    y_pred[:num_true] = local_train_losses

    if balanced:
        y_pred[num_true:] = test_losses[torch.randperm(test_losses.shape[0])[:num_true]]
    else:
        y_pred[num_true:] = test_losses

    # Use the balanced y_pred for the ROC curve
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    # print("Shapes: ", y_pred.shape, y_true.shape)

    # fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_true, y_pred)
    res = {
        # "fpr": fpr,
        # "tpr": tpr,
        # "thresholds": thresholds,
        "roc_auc": roc_auc,
    }
    return res


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


def load_model_from_path(model_path, model, shapes, lens, device=None):
    model_weigths = np.load(model_path)
    model.load_state_dict(deserialized_model(model_weigths, model, shapes, lens))
    if device is not None:
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
    res = {}
    threshold_result_unbalanced = threshold_attack(
        1 - losses_train, 1 - losses_test, balanced=False
    )
    res["roc_auc"] = threshold_result_unbalanced["roc_auc"]

    threshold_result_balanced = threshold_attack(
        1 - losses_train, 1 - losses_test, balanced=True
    )
    res["roc_auc_balanced"] = threshold_result_balanced["roc_auc"]
    return pd.DataFrame(res, index=[0])


def run_linkability_attack(
    running_model,
    model_path,
    expected_agent,
    attack_object: LinkabilityAttack,
    shapes,
    lens,
    device,
):
    load_model_from_path(
        model_path=model_path,
        model=running_model,
        shapes=shapes,
        lens=lens,
        device=device,
    )

    res = attack_object.log_all_losses(running_model)
    all_losses_list = [(agent, loss_agent) for agent, loss_agent in res.items()]
    all_losses_list.sort(key=lambda x: x[1])
    top_5 = [int(all_losses_list[i][0].split("_")[2]) for i in range(5)]

    res["linkability_top1"] = expected_agent == top_5[0]
    res["linkability_top1_guess"] = top_5[0]
    res["linkability_top5"] = expected_agent in top_5

    for index, (agent, _) in enumerate(all_losses_list):
        if expected_agent == agent:
            res["linkability_real_rank"] = index
            break

    return pd.DataFrame(res, index=[0])


def attack_experiment(
    experiment_df,
    results_directory: str,
    experiment_name: str,
    model_initialiser,
    batch_size,
    dataset,
    nb_agents,
    seed,
    kshards,
    device_type: str = "cpu",
    attack="threshold",
    loss_function=torch.nn.CrossEntropyLoss,
):
    assert attack in ["threshold", "linkability"]
    assert os.path.isdir(os.path.join(results_directory))
    results_file = os.path.join(results_directory, f"{attack}_{experiment_name}.csv")
    if os.path.exists(results_file):
        print(f"Attacks already computed for {experiment_name}, skipping")
        return

    device = torch.device(device_type)
    print(f"Attacking {experiment_name}:  ", end="")
    trainset_partitioner, testset = load_experiments.load_dataset_partitioner(
        dataset, nb_agents, seed, kshards
    )
    testset = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    res = pd.DataFrame({})
    current_experiment = experiment_df[
        experiment_df["experiment_name"] == experiment_name
    ]

    running_model = model_initialiser()
    shapes, lens = generate_shapes(running_model)

    loss = loss_function()
    attack_object = None
    if attack == "linkability":
        attack_object = LinkabilityAttack(
            num_clients=nb_agents,
            client_datasets=trainset_partitioner,
            loss=loss,
            eval_batch_size=batch_size,
            device=device,
        )

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
                f"Launching {attack} attack {agent}->{target} at iteration {iteration}"
                + " " * 10,
                end="\r",
            )
            if attack == "linkability":
                assert attack_object is not None
                attack_result = run_linkability_attack(
                    running_model=running_model,
                    model_path=model_path,
                    expected_agent=agent,  # We want to guess the CURRENT agent!
                    attack_object=attack_object,
                    shapes=shapes,
                    lens=lens,
                    device=device,
                )
            else:
                attack_result = run_threshold_attack(
                    running_model,
                    model_path,
                    trainset,
                    testset,
                    shapes,
                    lens,
                    device=device,
                )
            attack_result["iteration"] = iteration
            attack_result["agent"] = agent
            attack_result["target"] = target
            res = pd.concat([res, attack_result])
    # Rewrite the columns in a better order
    columns = res.columns.tolist()
    columns.sort(key=lambda x: (1, "") if "loss_trainset_" in x else (0, x))

    res = res[columns]
    # Save the file
    print(f"Writing results to {results_file}")
    res.to_csv(results_file)
    print(f"Finished attacking {experiment_name}")
    return


def main():
    attack = "linkability"
    batch_size = 512
    dataset = "CIFAR10"
    nb_agents = 128
    nb_machines = 8
    seed = 90
    kshards = 2
    model_architecture = LeNet
    device_type = "cuda"

    # ---------
    # Running the main attacks on all the experiments
    # ---------
    print("---- Starting main attacks ----")
    experiments_dir = "results/my_results/icml_experiments/cifar10/"
    results_path = (
        "results/my_results/icml_experiments/cifar10_attack_results_unbalanced/"
    )
    nb_processes = 20
    print("Loading experiments dirs")
    all_experiments_df = load_experiments.get_all_experiments_properties(
        experiments_dir, nb_agents, nb_machines
    )

    # When debugging, save the dataframe and load it to avoid cold starts.
    # all_experiments_df.to_csv("experiments_df.csv")
    # all_experiments_df = pd.read_csv("experiments_df.csv")

    # Use this we want to reduce the size of the experiments to consider for quick results
    # results_path = "results/my_results/icml_experiments/cifar10_attack_results_quick/"
    # all_experiments_df = all_experiments_df[
    #     all_experiments_df["iteration"].isin([500, 5000, 10000])
    # ]

    all_experiments_df.reset_index()

    futures = []
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=nb_processes, mp_context=context
    ) as executor:
        for experiment_name in sorted(pd.unique(all_experiments_df["experiment_name"])):
            # attack_experiment(
            #     copy.deepcopy(all_experiments_df),
            #     results_directory=copy.deepcopy(results_path),
            #     experiment_name=copy.deepcopy(experiment_name),
            #     model_initialiser=copy.deepcopy(model_architecture),
            #     batch_size=copy.deepcopy(batch_size),
            #     dataset=copy.deepcopy(dataset),
            #     nb_agents=copy.deepcopy(nb_agents),
            #     seed=copy.deepcopy(seed),
            #     kshards=copy.deepcopy(kshards),
            #     device_type=copy.deepcopy(device_type),
            #     attack=copy.deepcopy(attack),
            # )
            # break
            future = executor.submit(
                attack_experiment,
                copy.deepcopy(all_experiments_df),
                results_directory=copy.deepcopy(results_path),
                experiment_name=copy.deepcopy(experiment_name),
                model_initialiser=copy.deepcopy(model_architecture),
                batch_size=copy.deepcopy(batch_size),
                dataset=copy.deepcopy(dataset),
                nb_agents=copy.deepcopy(nb_agents),
                seed=copy.deepcopy(seed),
                kshards=copy.deepcopy(kshards),
                device_type=copy.deepcopy(device_type),
                attack=copy.deepcopy(attack),
            )
            futures.append(future)
        done, _ = concurrent.futures.wait(futures)
        results = []
        for future in done:
            results.append(future.result())

        print(results)


if __name__ == "__main__":
    main()
