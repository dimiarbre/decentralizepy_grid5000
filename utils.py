import copy
import os
from itertools import product
from typing import Optional, Type, TypeVar

from localconfig import LocalConfig


def parse(h):
    res = h.split(":")
    return [int(e) for e in res]


def to_hours(l):
    res = [str(e) for e in l]
    res = ["0" + e if len(e) == 1 else e for e in res]
    return ":".join(res)


def add_times(h1, h2):
    h1_parsed = parse(h1)
    h2_parsed = parse(h2)

    assert len(h1_parsed) == len(h2_parsed)

    res = []
    s = 0
    for i in range(len(h1_parsed) - 1, -1, -1):
        s += h1_parsed[i] + h2_parsed[i]
        if i > 0:
            res.append(s % 60)
            s = s // 60
        else:
            res.append(s)
    return to_hours(res[::-1])


def to_sec(walltime):
    l = parse(walltime)
    return l[-1] + 60 * (l[-2] + 60 * l[-3])


if __name__ == "__main__":
    WALLTIME = "9:59:00"
    ADDITIONNAL_TIME = "00:10:00"

    SUM_RESULT = add_times(WALLTIME, ADDITIONNAL_TIME)
    print(SUM_RESULT)
    print(to_sec(SUM_RESULT))


def read_ini(file_path: str, verbose=False) -> LocalConfig:
    """Function to load the dict configuration file.

    Args:
        file_path (str): The path to the config file
        verbose (bool, False): Whether to print extra information or not
    Returns:
        LocalConfig: The loaded configuration.
    """
    config = LocalConfig(file_path)
    if verbose:
        for section in config:
            print("Section: ", section)
            for key, value in config.items(section):
                print((key, value))
        print(dict(config.items("DATASET")))
    return config


model_estimation = {  # Model size in GB
    "RNET": 12 / 1000,  # 12 MB in GB.
    "LeNet": 360 / (1000 * 1000),  # 360 kB
    # NB: For MovieLens, the size can vary depending on the data split.
    "MatrixFactorization": 1000 / (1000 * 1000),  # 1000 kB.
}

T = TypeVar("T")


def safe_load(
    config: LocalConfig, section: str, parameter: str, expected_type: Type[T]
) -> T:
    value = config.get(section, parameter)
    if not isinstance(value, expected_type):
        raise ValueError(
            f"Invalid value for parameter {parameter}: expected {expected_type}, got {value}"
        )
    return value


def space_estimator(
    nb_experiments, nbnodes, total_iteration, config: LocalConfig
) -> int:
    save_frequency = safe_load(config, "SHARING", "save_models_for_attacks", int)
    save_all_models = safe_load(config, "SHARING", "save_all_models", int)
    model_name = safe_load(config, "DATASET", "model_class", str)
    if save_all_models:
        nb_nodes_saving = nbnodes
    else:
        nb_nodes_saving = config.get("SHARING", "nb_models_to_save")

    nb_models_to_save = (total_iteration // save_frequency) * nb_nodes_saving

    print(f"Estimated number of models to save: {nb_models_to_save}")
    experiment_estimation = nb_models_to_save * model_estimation[model_name]
    return experiment_estimation * nb_experiments


VARIANT_MAPPER = {  # The parameters for "sharing_package" and "sharing_class".
    "nonoise": ("decentralizepy.sharing.SharingAsymmetric", "SharingAsymmetric"),
    "muffliato": ("decentralizepy.sharing.Muffliato", "Muffliato"),
    "zerosum_selfnoise": ("decentralizepy.sharing.ZeroSumSharing", "ZeroSumSharing"),
}


def handle_special_parameters_values(
    config, parameter, parameter_value, variant, dataset
):
    if parameter == "topology":
        # This will be handled later when calling the right PeerSampler
        return True
    elif parameter == "variant":
        assert (
            parameter_value == variant
        )  # Sanity check, would be an error if it is not the case
        if ("any", dataset) not in baseconfig_mapping and (
            parameter_value,
            dataset,
        ) not in baseconfig_mapping:
            raise ValueError(
                f"Invalid parameter value for {parameter}: {parameter_value}"
            )
        sharing_package, sharing_class = VARIANT_MAPPER[parameter_value]
        config.set("SHARING", "sharing_package", sharing_package)
        config.set("SHARING", "sharing_class", sharing_class)
        if parameter_value in ["zerosum_selfnoise", "zerosum_noselfnoise"]:
            config.set("SHARING", "self_noise", parameter_value == "zerosum_selfnoise")

    elif parameter == "noise_level":
        if variant == "nonoise":
            raise ValueError(
                f"Should not have {parameter} for {variant}, but got {parameter_value}"
            )
        config.set("SHARING", "noise_std", noises_mapping[parameter_value])
    elif parameter == "random_seed":
        config.set("DATASET", "random_seed", int(parameter_value[4:]))
    elif parameter == "graph_degree":
        degree = int(parameter_value[6:])
        config.set("NODE", "graph_degree", degree)
    elif parameter == "lr":
        lr = float(parameter_value[2:])
        config.set("OPTIMIZER_PARAMS", "lr", lr)
    elif parameter == "rounds":
        nb_rounds = int(parameter_value[:-6])
        config.set("TRAIN_PARAMS", "rounds", nb_rounds)
    elif parameter == "batchsize":
        batch_size = int(parameter_value[9:])
        config.set("TRAIN_PARAMS", "batch_size", batch_size)
    else:
        return False
    return True


baseconfig_mapping = {
    ("any", "cifar"): os.path.join("run_configuration/only_training_nonoise.ini"),
    ("nonoise", "femnist"): os.path.join("run_configuration/femnist_nonoise.ini"),
    ("muffliato", "femnist"): os.path.join("run_configuration/femnist_muffliato.ini"),
    ("zerosum_selfnoise", "femnist"): os.path.join(
        "run_configuration/femnist_zerosum.ini"
    ),
    ("zerosum_noselfnoise", "femnist"): os.path.join(
        "run_configuration/femnist_zerosum_noselfnoise.ini"
    ),
    ("any", "femnistLabelSplit"): os.path.join(
        "run_configuration/femnist_labelsplit_nonoise.ini"
    ),
    ("any", "movielens"): os.path.join(
        "run_configuration/config_movielens_sharing.ini"
    ),
}

noises_mapping = {
    "0p25th": 0.225 / 0.25,
    "0p5th": 0.225 / 0.5,
    "0p75th": 0.225 / 0.75,
    "1th": 0.225,
    "2th": 0.1125,
    "2p5th": 0.225 / 2.5,
    "3th": 0.225 / 3,
    "3p5th": 0.225 / 3.5,
    "4th": 0.05625,
    "5th": 0.225 / 5,
    "6th": 0.225 / 6,
    "7th": 0.225 / 7,
    "8th": 0.028125,
    "16th": 0.0140625,
    "32th": 0.00703125,
    "64th": 0.003515625,
    "128th": 0.001757813,
}


def generate_config(combination_dict, dataset) -> Optional[LocalConfig]:
    variant = combination_dict["variant"]
    if ("any", dataset) in baseconfig_mapping:
        baseconfig = baseconfig_mapping[("any", dataset)]
    else:
        baseconfig = baseconfig_mapping[(variant, dataset)]
    config = read_ini(baseconfig)
    for parameter, parameter_value in combination_dict.items():
        is_handled = handle_special_parameters_values(
            config=config,
            parameter=parameter,
            parameter_value=parameter_value,
            variant=variant,
            dataset=dataset,
        )
        if is_handled is None:
            print(f"Skipping configuration {combination_dict}")
            return None
        if not is_handled:
            for section in config:
                for item, current_value in config.items(section):
                    if parameter == item:
                        config.set(section, parameter, parameter_value)
        else:
            print(f"Skipping {parameter} for {variant}")
    return config


def generate_config_files(
    attributes_values,
    dataset,
) -> dict[str, tuple[dict[str, str], LocalConfig]]:
    # TODO: handle dynamic topology
    configs = {}
    all_combinations: list[dict[str, str]] = []
    for variant in attributes_values["variant"]:
        current_attributes_dict = copy.deepcopy(attributes_values)
        del current_attributes_dict["variant"]
        if variant == "nonoise":
            del current_attributes_dict["noise_level"]

        attribute_names = list(current_attributes_dict.keys())

        # Generate all combinations
        attribute_values = list(current_attributes_dict.values())
        current_combinations = list(product(*attribute_values))
        all_combinations = all_combinations + [
            dict([("variant", variant)] + list(zip(attribute_names, combination)))
            for combination in current_combinations
        ]

    print(f"Generating {len(all_combinations)} combinations")
    # Print the combinations
    for combination_dict in all_combinations:
        print(f"Setting {combination_dict} in configuration")
        current_config = generate_config(combination_dict, dataset=dataset)
        if current_config is not None:
            configs[
                "_".join([param_value for _, param_value in combination_dict.items()])
            ] = (combination_dict, current_config)
        else:
            print(
                f"Config was skipped because of incompatible parameters: {combination_dict}"
            )
    return configs


def main():
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
    all_configs = generate_config_files(possible_attributes, "cifar")
    # print(all_configs)
    for name, config in all_configs.items():
        print(f"{name}")
        print(config[1])
        _ = input()
    return all_configs


if __name__ == "__main__":
    main()
