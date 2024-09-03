import argparse
import concurrent.futures
import copy
import json
import multiprocessing
import os
import sys
import time

from g5k_execution import launch_experiment
from utils import generate_config_files

g5kconfig_mapping: dict[tuple[str, str], str] = {
    ("nonoise", "cifar"): os.path.join("g5k_config/training_128nodes_nonoise.json"),
    ("muffliato", "cifar"): os.path.join("g5k_config/training_128nodes_muffliato.json"),
    ("zerosum_selfnoise", "cifar"): os.path.join(
        "g5k_config/training_128nodes_zerosum.json"
    ),
    ("zerosum_noselfnoise", "cifar"): os.path.join(
        "g5k_config/training_128nodes_zerosum_noselfnoise.json"
    ),
    # Femnist dataset
    ("nonoise", "femnist"): os.path.join(
        "g5k_config/femnist_128nodes_static_nonoise.json"
    ),
    ("muffliato", "femnist"): os.path.join(
        "g5k_config/femnist_128nodes_muffliato.json"
    ),
    ("zerosum_selfnoise", "femnist"): os.path.join(
        "g5k_config/femnist_128nodes_zerosum.json"
    ),
    ("zerosum_noselfnoise", "femnist"): os.path.join(
        "g5k_config/femnist_128nodes_zerosum_noselfnoise.json"
    ),
}

toplogy_dynamicity_mapping: dict[str, str] = {
    "static": "testingPeerSamplerMultipleAvgRounds.py",
    "dynamic": "testingPeerSamplerDynamicMultipleAvgRounds.py",
}


def space_estimator(nb_experiments, dataset):
    if dataset == "femnist":
        experiment_estimation = 20
    elif dataset == "cifar":
        experiment_estimation = 1.8
    else:
        raise ValueError(f"Unknown dataset type {dataset}")
    return experiment_estimation * nb_experiments


def launch_experiment_wrapper(
    g5k_config, decentralizepy_config, debug, is_remote, name
):
    print(f"Launching experiment {name}")
    log_file_path = "/tmp/logs/" + name + ".out"
    sys.stdout = open(log_file_path, "w+")
    print(
        f"Launching {g5k_config} and config {decentralizepy_config}, debug={debug} and is_remote={is_remote}"
    )
    launch_experiment(
        g5k_config=g5k_config,
        decentralizepy_config=decentralizepy_config,
        debug=debug,
        is_remote=is_remote,
    )
    os.remove(log_file_path)


def launch_batch(
    possible_attributes, is_remote, nb_workers=10, job_type=None, dataset="cifar"
):
    if job_type is None:
        job_type = []
    all_configs = generate_config_files(possible_attributes, dataset=dataset)
    # print(all_configs)
    for name, (attributes, config) in all_configs.items():
        print("---" * 15)
        print(f"{name}, {attributes}")
        print(config)
        # _ = input()
    for name, (attributes, config) in all_configs.items():
        print("---" * 15)
        print(f"{name}, {attributes}")
    nb_configs = len(all_configs)
    res = input(
        f"Are you sure you want to launch {nb_configs} experiments with {nb_workers} workers?\n"
        + f"This should take around {space_estimator(nb_configs,dataset=dataset):.2f} GB of space\ny/n -"
    )
    if res != "y":
        print("Aborting launchig experiments")
        return False
    print()
    futures = []
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=nb_workers, mp_context=context
    ) as executor:
        for name, (attributes, decentralizepy_config) in all_configs.items():
            toplogy_dynamicity = attributes["topology"]
            g5k_config_path = g5kconfig_mapping[(attributes["variant"], dataset)]
            with open(g5k_config_path) as g5k_config_file:
                g5k_config = json.load(g5k_config_file)

            g5k_config["job_name"] = f"{dataset}_{name}"
            g5k_config["EVAL_FILE"] = toplogy_dynamicity_mapping[toplogy_dynamicity]
            avgsteps_str = attributes["avgsteps"][:-8]
            assert (
                avgsteps_str.isnumeric
            ), f"{avgsteps_str} should be an int, total:{attributes['avgsteps']}"
            g5k_config["AVERAGING_STEPS"] = int(avgsteps_str)
            g5k_config["job_type"] = job_type
            # launch_experiment_wrapper(
            #     g5k_config=g5k_config,
            #     decentralizepy_config=decentralizepy_config,
            #     debug=False,
            #     is_remote=is_remote,
            #     name=copy.deepcopy(name),
            # )
            # break
            futures.append(
                executor.submit(
                    launch_experiment_wrapper,
                    g5k_config=copy.deepcopy(g5k_config),
                    decentralizepy_config=copy.deepcopy(decentralizepy_config),
                    debug=copy.deepcopy(False),
                    is_remote=copy.deepcopy(is_remote),
                    name=copy.deepcopy(name),
                )
            )
            # Wait a bit to increase the chance for the reservations to be in the same order
            time.sleep(2)
    print("Finished all workers!")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Parser for launching batch experiments."
    )
    parser.add_argument(
        "--is_remote",
        action="store_true",
        help="Flag indicating if the job is executed remotely or locally.",
    )
    parser.add_argument(
        "--job_type",
        choices=["night", "day"],
        default=[[]],
        help="Type of the job.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar", "femnist"],
        default="cifar",
        help="Dataset name or path",
    )

    parser.add_argument(
        "--nb_workers",
        type=int,
        default=20,
        help="Number of threads that launches attacks. Also the number of simultaneous jobs.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    possible_attributes = {
        "nbnodes": ["128nodes"],
        #
        # "variant": ["nonoise", "zerosum_selfnoise", "zerosum_noselfnoise"],
        # "variant": ["nonoise", "zerosum_selfnoise"],
        # "variant": ["zerosum_selfnoise"],
        "variant": ["nonoise"],
        # "variant": ["muffliato"],
        #
        # "avgsteps": ["10avgsteps"],
        # "avgsteps": ["1avgsteps"],
        "avgsteps": [
            "1avgsteps",
            # "5avgsteps",
            # "10avgsteps",
            # "15avgsteps",
            # "20avgsteps",
        ],
        #
        "noise_level": ["128th", "64th", "32th", "16th", "8th", "4th", "2th", "1th"],
        # "noise_level": ["128th", "1th"],
        # "noise_level": ["0p75th"],
        # "noise_level": ["2p5th", "3th", "3p5th", "5th", "6th", "7th"],
        # "noise_level": ["2p5th", "3th", "3p5th", "5th", "6th", "7th"],
        # "noise_level": ["0p25th", "0p5th", "0p75th", "2p5th", "3th", "3p5th"],
        # "noise_level": ["4th", "16th", "64th"],
        #
        # "topology": ["static", "dynamic"],
        "topology": ["static"],
        # "topology": ["dynamic"],
        #
        # "random_seed": [f"seed{i}" for i in range(91, 106)],
        "random_seed": ["seed90"],
        #
        "graph_degree": ["degree6"],
        #
        # "model_class": ["LeNet"],
        "model_class": ["RNET"],
        # "model_class": ["CNN"],
        #
        "lr": ["lr0.05", "lr0.01", "lr0.10"],
        # "lr": ["lr0.01"],
        #
        "rounds": ["3rounds", "1rounds"],
        # "rounds":["3rounds"],
    }
    ARGS = parse_arguments()
    NB_WORKERS = ARGS.nb_workers
    IS_REMOTE = ARGS.is_remote
    job_type = ARGS.job_type
    DATASET = ARGS.dataset

    print(f"IS_REMOTE: {IS_REMOTE}, DATASET: {DATASET}, job_type: {job_type}")
    launch_batch(
        possible_attributes=possible_attributes,
        is_remote=IS_REMOTE,
        nb_workers=NB_WORKERS,
        job_type=job_type,
        dataset=DATASET,
    )
