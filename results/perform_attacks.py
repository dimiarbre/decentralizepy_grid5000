import argparse
import concurrent.futures
import copy
import functools
import heapq
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import traceback
from typing import Optional

import load_experiments
import numpy as np
import pandas as pd
import torch
from LinkabilityAttack import LinkabilityAttack
from load_experiments import ALL_ATTACKS, POSSIBLE_MODELS
from RocPlotter import RocPlotter
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


def get_biased_threshold_acc(train_losses, test_losses, top_kth: list[int]):

    max_k = max(top_kth)
    paired_train = [(loss, 1) for loss in train_losses]
    paired_test = [(loss, 0) for loss in test_losses]

    merged_list = paired_train + paired_test

    if max_k > len(paired_train):
        print(
            f"\n---------\nWARNING: using window of size {max_k} when the training set only has {len(paired_train)} elements!\n-------"
        )
    # Remove elements when the window is bigger than the
    fixed_top_kth = [k for k in top_kth if k <= len(merged_list)]

    max_k = max(fixed_top_kth)

    lowest_losses_max_k = heapq.nsmallest(max_k, merged_list)

    res = {}
    for k in top_kth:
        window = lowest_losses_max_k[:k]
        nb_success = sum([origin for (_, origin) in window])
        success_rate = nb_success / k
        assert success_rate <= 1
        res[f"top{k}_acc"] = success_rate

    return res


def threshold_attack(
    local_train_losses,
    test_losses,
    balanced=False,
    plotter: Optional[RocPlotter] = None,
):
    # We need the losses to be increasing the more likely we are to be in the train set,
    # so 1-loss should be given as argument.
    num_true = local_train_losses.shape[0]
    # print("Number of training samples: ", num_true)

    assert (
        num_true <= test_losses.shape[0]
    ), f"Not enough test elements: {test_losses.shape[0]} when at least {num_true} where expected"

    if balanced:
        y_true = torch.ones((num_true + num_true,), dtype=torch.int32)
        y_pred = torch.zeros((num_true + num_true,), dtype=torch.float32)
    else:
        y_true = torch.ones((num_true + test_losses.shape[0]), dtype=torch.int32)
        y_pred = torch.zeros((num_true + test_losses.shape[0]), dtype=torch.float32)

    y_true[:num_true] = 0
    y_pred[:num_true] = local_train_losses

    if balanced:
        y_pred[num_true:] = test_losses[torch.randperm(test_losses.shape[0])[:num_true]]
    else:
        y_pred[num_true:] = test_losses

    # Use the balanced y_pred for the ROC curve
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    # print("Shapes: ", y_pred.shape, y_true.shape)

    roc_auc = roc_auc_score(y_true, y_pred)

    if plotter is not None:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
        plotter.plot_roc(
            fpr,
            tpr,
            thresholds,
            roc_auc,
            losses_train=y_pred[:num_true],
            losses_test=y_pred[num_true:],
        )

    res = {
        # "fpr": fpr,
        # "tpr": tpr,
        # "thresholds": thresholds,
        "roc_auc": roc_auc,
    }
    return res


def generate_shapes(model):
    shapes = []
    lens = []
    with torch.no_grad():
        for _, v in model.state_dict().items():
            shapes.append(v.shape)
            t = v.flatten().numpy()
            lens.append(t.shape[0])
    return shapes, lens


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
    attack="threshold",
    plotter_unbalanced: Optional[RocPlotter] = None,
    plotter_balanced: Optional[RocPlotter] = None,
):
    both_attacks = False
    default_res_threshold = {
        "roc_auc": torch.nan,
        "roc_auc_balanced": torch.nan,
    }
    default_res_biasedthreshold = {f"top{k}-acc": torch.nan for k in ALL_TOPK_ATTACKS}
    if "+" in attack:
        assert attack == "threshold+biasedthreshold"
        both_attacks = True
    elif attack == "threshold" or attack == "biasedthreshold":
        both_attacks = False
    else:
        raise ValueError(f"Unknown attack {attack}")

    load_experiments.load_model_from_path(
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

        return {
            "threshold": pd.DataFrame(default_res_threshold, index=[0]),
            "biasedthreshold": pd.DataFrame(default_res_biasedthreshold, index=[0]),
        }

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
    res_threshold = {}
    if both_attacks or attack == "threshold":
        threshold_result_unbalanced = threshold_attack(
            losses_train, losses_test, balanced=False, plotter=plotter_unbalanced
        )
        res_threshold["roc_auc"] = threshold_result_unbalanced["roc_auc"]

        threshold_result_balanced = threshold_attack(
            losses_train, losses_test, balanced=True, plotter=plotter_balanced
        )
        res_threshold["roc_auc_balanced"] = threshold_result_balanced["roc_auc"]

    res_biasedthreshold = {}
    if both_attacks or attack == "biasedthreshold":
        res_biasedthreshold = get_biased_threshold_acc(
            train_losses=losses_train,
            test_losses=losses_test,
            top_kth=ALL_TOPK_ATTACKS,
        )
        # TODO: make a biasedthreshold on the balanced losses?
        # TODO: investigate the small success of this attack.

    return {
        "threshold": pd.DataFrame(res_threshold, index=[0]),
        "biasedthreshold": pd.DataFrame(res_biasedthreshold, index=[0]),
    }


def run_linkability_attack(
    running_model,
    model_path,
    expected_agent,
    attack_object: LinkabilityAttack,
    shapes,
    lens,
    device,
):
    load_experiments.load_model_from_path(
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
    attack_todo="threshold",
    loss_function=torch.nn.CrossEntropyLoss,
    debug=False,
):
    # Load this experiment's configuration file
    config_file = os.path.join(experiment_path, "config.ini")
    assert os.path.exists(config_file), f"Cannot find config file at {config_file}"
    config = load_experiments.read_ini(config_file)

    dataset = config.dataset.dataset_class

    seed = load_experiments.safe_load_int(config, "DATASET", "random_seed")
    if dataset in ["Femnist", "FemnistLabelSplit"]:
        kshards = None
    else:
        kshards = load_experiments.safe_load_int(config, "DATASET", "shards")

    model_name = config.dataset.model_class
    model_initialiser = POSSIBLE_MODELS[model_name]

    assert attack_todo in [
        "threshold",
        "linkability",
        "biasedthreshold",
        "threshold+biasedthreshold",  # Those two attacks can be combined to save time and energy.
    ]

    if attack_todo == "threshold+biasedthreshold":
        results_files = {}
        result_file_threshold = os.path.join(
            experiment_path, f"threshold_{experiment_name}.csv"
        )
        if os.path.exists(result_file_threshold):
            print(
                f"threshold-attack already computed for {experiment_name}, not computing"
            )
            attack_todo = "biasedthreshold"
        else:
            results_files["threshold"] = result_file_threshold
        result_file_biasedthreshold = os.path.join(
            experiment_path, f"biasedthreshold_{experiment_name}.csv"
        )
        if os.path.exists(result_file_threshold):
            print(
                f"threshold-attack already computed for {experiment_name}, not computing"
            )
            if attack_todo == "biasedthreshold":
                print(f"Both attacks already computed for {experiment_name}, skipping")
                return
            attack_todo = "threshold"
        else:
            results_files["biasedthreshold"] = result_file_biasedthreshold

    if attack_todo in ["linkability", "threshold", "biasedthreshold"]:
        results_files = {}
        both_attacks = False
        results_file = os.path.join(
            experiment_path, f"{attack_todo}_{experiment_name}.csv"
        )
        if os.path.exists(results_file):
            print(
                f"{attack_todo}-attack already computed for {experiment_name}, skipping"
            )
            return
        results_files[attack_todo] = [results_file]

    device = torch.device(device_type)
    print(f"Launching {attack_todo}-attack on {experiment_name}.")
    t1 = time.time()

    trainset_partitioner, testset = load_experiments.load_dataset_partitioner(
        dataset, nb_agents, seed, kshards, debug=DEBUG
    )
    testset = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    current_experiment = experiment_df[
        experiment_df["experiment_name"] == experiment_name
    ]

    running_model = model_initialiser()
    shapes, lens = generate_shapes(running_model)

    loss = loss_function()
    attack_object = None
    # This is a trick to save time: only have one attack instance for all the experiment
    # instead of creating a new one for each model in the experiment.
    # Avoids loading repeatedly the dataset.
    if attack_todo == "linkability":
        attack_object = LinkabilityAttack(
            num_clients=nb_agents,
            client_datasets=trainset_partitioner,
            loss=loss,
            eval_batch_size=batch_size,
            device=device,
        )

    total_result = {}
    agent_list = sorted(pd.unique(current_experiment["agent"]))
    for agent in agent_list:
        if agent == 0:
            roc_plotter_unbalanced = RocPlotter(
                os.path.join(experiment_path, f"unbalanced_roc_loss_distrib.pdf"),
                nb_columns=5,
            )
            roc_plotter_balanced = RocPlotter(
                os.path.join(experiment_path, f"balanced_roc_loss_distrib.pdf"),
                nb_columns=5,
            )
        else:
            roc_plotter_unbalanced = None
        current_agent_experiments = current_experiment[
            current_experiment["agent"] == agent
        ]
        trainset = trainset_partitioner.use(agent)
        trainset = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=False
        )

        rows = [row for _, row in current_agent_experiments.iterrows()]
        rows = sorted(rows, key=lambda row: int(row["iteration"]))

        for row in rows:
            results = {}
            iteration = row["iteration"]
            model_path = row["file"]
            target = row["target"]
            tcurrent = time.time()
            if debug:
                endline = "\n"
            else:
                endline = "\r"
            print(
                f"Launching {attack_todo} attack {agent}->{target} at iteration {iteration}, {(tcurrent-t1)/60:.2f} minutes taken to reach this point"
                + " " * 10,
                end=endline,
            )
            if attack_todo == "linkability":
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
                results[attack] = [attack_result]

            else:
                if roc_plotter_unbalanced is not None:
                    roc_plotter_unbalanced.set_next_plot(
                        iteration=iteration,
                        log_next=iteration in [1, 100, 1000, 2500, 4000, 5000],
                    )
                if roc_plotter_balanced is not None:
                    roc_plotter_balanced.set_next_plot(
                        iteration=iteration,
                        log_next=iteration in [1, 100, 1000, 2500, 4000, 5000],
                    )
                threshold_results = run_threshold_attack(
                    running_model,
                    model_path,
                    trainset,
                    testset,
                    shapes,
                    lens,
                    device=device,
                    debug=debug,
                    debug_name=experiment_name,
                    attack=attack_todo,
                    plotter_unbalanced=roc_plotter_unbalanced,
                    plotter_balanced=roc_plotter_balanced,
                )
                # Only save the necessary results.
                for attack_to_save in results_files:
                    results[attack_to_save] = threshold_results[attack_to_save]

            for attack, result in results.items():
                result["iteration"] = iteration
                result["agent"] = agent
                result["target"] = target
                results[attack] = result
                if debug:
                    print(f"it:{iteration} -  {agent}->{target}. {attack}:\n{result}\n")
                if attack not in total_result:
                    total_result[attack] = pd.DataFrame({})
                total_result[attack] = pd.concat(
                    [total_result[attack], results[attack]]
                )

    # save all the attacks
    for current_attack, result_file in results_files.items():
        res = total_result[current_attack]
        # Rewrite the columns in a better order
        columns = res.columns.tolist()
        columns.sort(key=lambda x: (1, "") if "loss_trainset_" in x else (0, x))
        if current_attack == "biasedthreshold":
            columns.sort(
                key=lambda x: (1, len(x), x) if "_acc" in x else (0, len(x), x)
            )

        res = res[columns]
        if res.shape[0] < 10:
            print(
                f"Only {res.shape[0]} in the result for {current_attack} - {experiment_name}"
            )
            print(res)
            raise ValueError(res)
        # Save the file
        print(f"Writing results to {result_file}")
        res.to_csv(result_file)
    t2 = time.time()
    print(
        f"Finished {attack_todo} attack on {experiment_name} in {(t2-t1)/3600:.2f}hours"
    )
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
    attacks=ALL_ATTACKS,
):
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
                        attack_todo=copy.deepcopy(attacks[0]),
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
                    attack_todo=copy.deepcopy(attack),
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


ALL_TOPK_ATTACKS = [1, 5, 10, 100, 250, 500]
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
    ALL_ATTACKS = ["threshold+biasedthreshold"]

    main(
        experiments_dir=EXPERIMENT_DIR,
        should_clear=SHOULD_CLEAR,
        nb_processes=NB_WORKERS,
        batch_size=BATCH_SIZE,
        nb_agents=NB_AGENTS,
        nb_machines=NB_MACHINES,
        attacks=ALL_ATTACKS,
    )
