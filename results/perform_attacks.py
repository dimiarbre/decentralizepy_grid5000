import argparse
import concurrent.futures
import copy
import functools
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import traceback

import load_experiments
import numpy as np
import pandas as pd
import torch
from LinkabilityAttack import LinkabilityAttack
from load_experiments import ALL_ATTACKS, POSSIBLE_MODELS
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from decentralizepy.datasets.CIFAR10 import LeNet

DEBUG = False


def error_catching_wrapper(func):
    @functools.wraps(func)
    def wrapped_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error occurred in function '{func.__name__}': {e}")
            traceback.print_exc()
            raise e

    return wrapped_function


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
    debug=False,
):
    losses = torch.tensor([])
    # Fixes inconsistent batch size
    # See https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/262741
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_predicted = 0
        for x, y in dataset:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_function(y_pred, y)
            loss = loss.to("cpu")
            losses = torch.cat([losses, loss])
            _, predictions = torch.max(y_pred, 1)
            for label, prediction in zip(y, predictions):
                if label == prediction:
                    total_correct += 1
                total_predicted += 1
    accuracy = total_correct / total_predicted
    if debug:
        print(f"Accuracy: {accuracy*100:.2f}%")
    return (losses, accuracy)


def run_threshold_attack(
    running_model,
    model_path,
    trainset,
    testset,
    shapes,
    lens,
    device=torch.device("cpu"),
    debug=False,
    debug_name="",
):
    load_model_from_path(
        model_path=model_path,
        model=running_model,
        shapes=shapes,
        lens=lens,
        device=device,
    )
    losses_train, acc_train = generate_losses(
        running_model, trainset, device=device, debug=debug
    )
    if losses_train.isnan().any():
        losses_train_nonan = losses_train[~losses_train.isnan()]
        percent_fail = (
            (len(losses_train) - len(losses_train_nonan)) / len(losses_train) * 100
        )
        print(
            f"{debug_name} - Found NaNs in train loss! Removed {percent_fail:.2f}% of train losses"
        )
        losses_train = losses_train_nonan
    losses_test, acc_test = generate_losses(
        running_model, testset, device=device, debug=debug
    )
    if losses_test.isnan().any():
        losses_test_nonan = losses_test[~losses_test.isnan()]
        percent_fail = (
            (len(losses_test) - len(losses_test_nonan)) / len(losses_test) * 100
        )
        print(
            f"{debug_name} - Found NaNs in test loss! Removed {percent_fail:.2f}% of test losses"
        )
        losses_test = losses_test_nonan
    if len(losses_test) == 0 or len(losses_train) == 0:
        # print(
        #     "Found a losses tensor of size 0, found lengths -"
        #     + f" train:{len(losses_train)} - test:{len(losses_test)}"
        # )
        res = {
            "roc_auc": torch.nan,
            "roc_auc_balanced": torch.nan,
        }
        return pd.DataFrame(res, index=[0])

    print(
        f"{debug_name} - Train accuracy {acc_train*100:.2f}, Test accuracy {acc_test*100:.2f} "
    )

    if debug:
        print(
            f"Train losses - avg:{losses_train.mean()}, std:{losses_train.std()}, min:{losses_train.min()}, max:{losses_train.max()}."
        )
        print(
            f"Test losses - avg:{losses_test.mean()}, std:{losses_test.std()}, min:{losses_test.min()}, max:{losses_test.max()}."
        )
    # We use 1 - losses to have an AUC>0.5 (works for CrossEntropyLoss)
    res = {}
    threshold_result_unbalanced = threshold_attack(
        -losses_train, -losses_test, balanced=False
    )
    res["roc_auc"] = threshold_result_unbalanced["roc_auc"]

    threshold_result_balanced = threshold_attack(
        -losses_train, -losses_test, balanced=True
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
    all_losses_list = [
        (int(agent.split("_")[2]), loss_agent) for agent, loss_agent in res.items()
    ]
    all_losses_list.sort(key=lambda x: x[1])
    top_5 = [all_losses_list[i][0] for i in range(5)]

    res["linkability_top1"] = expected_agent == top_5[0]
    res["linkability_top1_guess"] = top_5[0]
    res["linkability_top5"] = expected_agent in top_5

    for index, (agent, _) in enumerate(all_losses_list):
        if expected_agent == agent:
            res["linkability_real_rank"] = index
            break
    if "linkability_real_rank" not in res:
        print(
            f"ERROR: no linkability real rank found for {model_path} when it should have."
        )
        res["linkability_real_rank"] = np.nan
    return pd.DataFrame(res, index=[0])


@error_catching_wrapper
def attack_experiment(
    experiment_df,
    experiment_name: str,
    batch_size: int,
    nb_agents: int,
    experiment_path,
    device_type: str = "cpu",
    attack="threshold",
    loss_function=torch.nn.CrossEntropyLoss,
    debug=False,
):
    # Load this experiment's configuration file
    config_file = os.path.join(experiment_path, "config.ini")
    assert os.path.exists(config_file), f"Cannot find config file at {config_file}"
    config = load_experiments.read_ini(config_file)

    dataset = config.dataset.dataset_class

    seed = load_experiments.safe_load_int(config, "DATASET", "random_seed")
    if dataset == "Femnist" or dataset == "FemnistLabelSplit":
        kshards = None
    else:
        kshards = load_experiments.safe_load_int(config, "DATASET", "shards")

    model_name = config.dataset.model_class
    model_initialiser = POSSIBLE_MODELS[model_name]

    assert attack in ["threshold", "linkability"]
    results_file = os.path.join(experiment_path, f"{attack}_{experiment_name}.csv")
    if os.path.exists(results_file):
        print(f"{attack}-attack already computed for {experiment_name}, skipping")
        return

    device = torch.device(device_type)
    print(f"Launching {attack}-attack on {experiment_name}.")
    t1 = time.time()

    trainset_partitioner, testset = load_experiments.load_dataset_partitioner(
        dataset, nb_agents, seed, kshards, debug=DEBUG
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
            tcurrent = time.time()
            if debug:
                endline = "\n"
            else:
                endline = "\r"
            print(
                f"Launching {attack} attack {agent}->{target} at iteration {iteration}, {(tcurrent-t1)/60:.2f} minutes taken to reach this point"
                + " " * 10,
                end=endline,
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
                    debug=debug,
                    debug_name=experiment_name,
                )
                if debug:
                    print(
                        f"it:{iteration} -  {agent}->{target}. AUC={attack_result['roc_auc']}, Balanced AUC={attack_result['roc_auc_balanced']}  "
                    )
            attack_result["iteration"] = iteration
            attack_result["agent"] = agent
            attack_result["target"] = target
            res = pd.concat([res, attack_result])
    # Rewrite the columns in a better order
    columns = res.columns.tolist()
    columns.sort(key=lambda x: (1, "") if "loss_trainset_" in x else (0, x))

    res = res[columns]
    if res.shape[0] < 10:
        print(f"Only {res.shape[0]} in the result for {attack} - {experiment_name}")
        print(res)
        raise ValueError(res)
    # Save the file
    print(f"Writing results to {results_file}")
    res.to_csv(results_file)
    t2 = time.time()
    print(f"Finished {attack} attack on {experiment_name} in {(t2-t1)/3600:.2f}hours")
    return


def clear_models(experiment_name: str, experiment_path: str):
    for attack in ALL_ATTACKS:
        results_file = os.path.join(experiment_path, f"{attack}_{experiment_name}.csv")
        # Warning: this must be the same thing as is done in the model formating
        if not os.path.exists(results_file):
            print(f"Missing {attack} for experiment {experiment_name}, not clearing")
            return False
    cwd = subprocess.run(["pwd"], check=True)
    print(f"Current directory: {cwd}")
    print(f"All attacks computed for {experiment_name}: cleaning results")

    for folder in os.listdir(experiment_path):
        folder_path = os.path.join(experiment_path, folder)
        if os.path.isdir(folder_path):
            attacked_models_path = os.path.join(folder_path, "attacked_model")
            if os.path.isdir(attacked_models_path):
                print(f"Removing {attacked_models_path}")
                shutil.rmtree(attacked_models_path, ignore_errors=True)

    # files_to_remove = subprocess.run(
    #     ["find", experiment_path, "-type", "f", "-name", "*.npy"],
    #     check=True,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    # )
    # for file_byte in files_to_remove.stdout.splitlines():
    #     file = file_byte.decode("utf-8")
    #     assert os.path.isfile(file), f"Files should exist: {file}"
    #     os.remove(file)

    return True


def print_finished():
    n = 40
    print("-" * n + "\nAll attacks have been launched!\n" + "-" * n)


def main(
    experiments_dir,
    should_clear,
    nb_processes=10,
    batch_size=256,
    nb_agents=128,
    nb_machines=8,
):
    attacks = ALL_ATTACKS
    device_type = "cuda:0"

    # ---------
    # Running the main attacks on all the experiments
    # ---------
    print("---- Starting main attacks ----")

    # experiments_dir = "results/my_results/icml_experiments/cifar10/"
    # experiments_dir = "results/my_results/icml_experiments/additional_cifar10/"

    print("Loading experiments dirs")
    all_experiments_df = load_experiments.get_all_experiments_properties(
        experiments_dir, nb_agents, nb_machines, attacks
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
    print(all_experiments_df)
    print("Loading done, starting all attacks\n")
    futures = []
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=nb_processes, mp_context=context
    ) as executor:
        for attack in attacks:
            for experiment_name in sorted(
                pd.unique(all_experiments_df["experiment_name"])
            ):
                if DEBUG:
                    attack_experiment(
                        copy.deepcopy(all_experiments_df),
                        experiment_name=copy.deepcopy(experiment_name),
                        batch_size=copy.deepcopy(batch_size),
                        nb_agents=copy.deepcopy(nb_agents),
                        experiment_path=os.path.join(experiments_dir, experiment_name),
                        device_type=copy.deepcopy(device_type),
                        attack=copy.deepcopy(attacks[0]),
                        debug=DEBUG,
                    )
                    raise RuntimeError()
                future = executor.submit(
                    attack_experiment,
                    copy.deepcopy(all_experiments_df),
                    experiment_name=copy.deepcopy(experiment_name),
                    batch_size=copy.deepcopy(batch_size),
                    nb_agents=copy.deepcopy(nb_agents),
                    experiment_path=os.path.join(experiments_dir, experiment_name),
                    device_type=copy.deepcopy(device_type),
                    attack=copy.deepcopy(attack),
                    debug=DEBUG,
                )
                futures.append(future)
        executor.submit(print_finished)  # To have a display when things are finished
        done, _ = concurrent.futures.wait(futures)
        results = []
        for future in done:
            results.append(future.result())

        print(results)
    if should_clear:
        print("Clearing results:")
        for experiment_name in sorted(pd.unique(all_experiments_df["experiment_name"])):
            clear_models(
                experiment_name=experiment_name,
                experiment_path=os.path.join(experiments_dir, experiment_name),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Attacks saved models after simulation"
    )

    parser.add_argument(
        "experiment_dir",
        type=str,
        default="results/my_results/test/testing_femnist_convergence_rates/",
        help="Path to the experiment directory",
    )

    parser.add_argument(
        "--nb_workers",
        type=int,
        default=20,
        help="Number of workers to use (default: 20)",
    )

    parser.add_argument(
        "--clear", action="store_true", help="Whether to clear something"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size used for attacks. Scale this for faster attacks. Default 256.",
    )

    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Run in debug mode if true. Adds some printing.",
    )

    parser.add_argument(
        "--nb_agents",
        type=int,
        default=128,
        help="Number of simulated nodes in these attacks. Default 128.",
    )
    parser.add_argument(
        "--nb_machines",
        type=int,
        default=8,
        help="Number of physical machines used for the attacks. Default 8.",
    )

    args = parser.parse_args()

    EXPERIMENT_DIR = args.experiment_dir
    NB_WORKERS = args.nb_workers
    SHOULD_CLEAR = args.clear
    BATCH_SIZE = args.batch_size
    DEBUG = args.debug
    NB_AGENTS = args.nb_agents
    NB_MACHINES = args.nb_machines
    ALL_ATTACKS = ["threshold"]

    main(
        experiments_dir=EXPERIMENT_DIR,
        should_clear=SHOULD_CLEAR,
        nb_processes=NB_WORKERS,
        batch_size=BATCH_SIZE,
        nb_agents=NB_AGENTS,
        nb_machines=NB_MACHINES,
    )
