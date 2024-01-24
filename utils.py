import copy
import os
from itertools import product

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


def handle_special_parameters_values(config, parameter, parameter_value, variant):
    if parameter == "topology":
        # This will be handled later when calling the right PeerSampler
        return True
    elif parameter == "variant":
        assert (
            parameter_value == variant
        )  # Sanity check, would be an error if it is not the case
        if parameter_value not in baseconfig_mapping:
            raise ValueError(
                f"Invalid parameter value for {parameter}: {parameter_value}"
            )
        return True
    elif parameter == "noise_level":
        if variant == "nonoise":
            raise ValueError(
                f"Should not have {parameter} for {variant}, but got {parameter_value}"
            )
        config.set("SHARING", "noise_std", noises_mapping[parameter_value])
        return True
    elif parameter == "random_seed":
        config.set("DATASET", "random_seed", int(parameter_value[4:]))
        return True
    return False


baseconfig_mapping = {
    "nonoise": os.path.join("run_configuration/only_training_nonoise.ini"),
    "muffliato": os.path.join("run_configuration/only_training_muffliato.ini"),
    "zerosum_selfnoise": os.path.join("run_configuration/only_training_zerosum.ini"),
    "zerosum_noselfnoise": os.path.join(
        "run_configuration/only_training_zerosum_noselfnoise.ini"
    ),
}

noises_mapping = {
    "1th": 0.225,
    "2th": 0.1125,
    "4th": 0.05625,
    "8th": 0.028125,
    "16th": 0.0140625,
    "32th": 0.00703125,
    "64th": 0.003515625,
    "128th": 0.001757813,
}


def generate_config(combination_dict):
    variant = combination_dict["variant"]
    config = read_ini(baseconfig_mapping[variant])
    for parameter, parameter_value in combination_dict.items():
        is_handled = handle_special_parameters_values(
            config=config,
            parameter=parameter,
            parameter_value=parameter_value,
            variant=variant,
        )
        if is_handled is None:
            print(f"Skipping configuration {combination_dict}")
            return None
        if not is_handled:
            for section in config:
                if parameter in config.items(section):
                    config.set(section, parameter, parameter_value)
        else:
            print(f"Skipping {parameter} for {variant}")
    return config


def generate_config_files(
    attributes_values,
) -> dict[str, tuple[dict[str, str], LocalConfig]]:
    # TODO: handle dynamic topology
    configs = {}
    all_combinations = []
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
        current_config = generate_config(combination_dict)
        if current_config is not None:
            configs[
                "_".join([param_value for _, param_value in combination_dict.items()])
            ] = (combination_dict, str(current_config))
        else:
            print(
                f"Config was skipped because of incompatible parameters: {combination_dict}"
            )
    return configs


def main():
    possible_attributes = {
        "variant": ["nonoise", "muffliato", "zerosum", "zerosum_noselfnoise"],
        "noise_level": ["64th", "32th", "16th", "8th", "4th", "2th"],
        # "topology": ["static", "dynamic"],
        "random_seed": ["seed91", "seed92", "seed93"],
    }
    all_configs = generate_config_files(possible_attributes)
    # print(all_configs)
    for name, config in all_configs.items():
        print(f"{name}")
    return all_configs


if __name__ == "__main__":
    main()
