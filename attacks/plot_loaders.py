import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from load_experiments import read_ini, safe_load
from localconfig import LocalConfig

ATTRIBUTE_DICT = {
    "network_size": ["128nodes"],
    "topology_type": ["static", "dynamic"],
    "variant": ["nonoise", "muffliato", "zerosum"],
    "avgsteps": ["20avgsteps", "15avgsteps", "10avgsteps", "5avgsteps", "1avgsteps"],
    "additional_attribute": ["selfnoise", "noselfnoise", "nonoise", "muffliato"],
    "noise_level": [
        "nonoise",
        "0p25th",
        "0p5th",
        "0p75th",
        "1th",
        "2th",
        "2p5th",
        "3th",
        "3p5th",
        "4th",
        "5th",
        "6th",
        "7th",
        "8th",
        "16th",
        "32th",
        "64th",
        "128th",
    ],
    "lr": ["lr0.05", "lr0.01", "lr0.10"],
    "local_rounds": ["1rounds", "3rounds"],
    "batch_size": ["batch64"],
    "seed": [f"seed{val}" for val in range(90, 106)],
    "model": ["LeNet", "CNN", "RNET"],
}

NOISES_MAPPING = {
    "nonoise": 0,
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

# Inverse the noise definition to make it easier for the readers - 2⁰σ, 2¹σ, 2²σ,....
NOISE_MAPPING_LOG = {
    "nonoise": "",
    "0p25th": 9,
    "0p5th": 8,
    "0p75th": 7.4,
    "1th": 7,
    "2th": 6,
    "2p5th": 5.7,
    "3th": 5.5,
    "3p5th": 5.2,
    "4th": 5,
    "5th": 4.75,
    "6th": 4.5,
    "7th": 4.25,
    "8th": 4,
    "16th": 3,
    "32th": 2,
    "64th": 1,
    "128th": 0,
}


def percentile(n):
    """Function to create the percentile aggregator.
    Taken from https://stackoverflow.com/questions/17578115/pass-percentiles-to-pandas-agg-function

    Args:
        n (float): The percentage of the percentile. Can be 0.5,0.95, ...
    """

    def percentile_(x):
        return x.quantile(n)

    percentile_.__name__ = "percentile_{:02.0f}".format(100 * n)
    return percentile_


def count_percentage_success(column: pd.Series):
    counts = column.value_counts(normalize=True)
    if True not in counts:
        return np.nan
    res = counts[True]
    return res


def start_avg(column: pd.Series):
    res = []
    running_elements: pd.Series = pd.Series({})
    for index, value in column.items():
        if np.isnan(value):
            res.append(np.NaN)
        elif running_elements.empty:
            running_elements = pd.Series(value)
            res.append(value)
        else:
            running_elements = pd.concat([running_elements, pd.Series(value)])
            res.append(running_elements.mean())
    return res


def get_attributes(filename, attribute_dict=ATTRIBUTE_DICT):
    parsed_filename = filename.split("_")
    res_attribute = {}
    for global_attribute, possible_values in attribute_dict.items():
        for current_attribute in parsed_filename:
            if current_attribute in possible_values:
                if current_attribute not in res_attribute:
                    res_attribute[global_attribute] = current_attribute
                else:
                    raise RuntimeError(f"Duplicate of attributes in {filename}")

        if global_attribute not in res_attribute:
            # print(f"No value found for attribute {global_attribute} for {filename}")
            res_attribute[global_attribute] = None
    return res_attribute


def check_attribute(experiment_attributes, attributes_to_check):
    for attribute, expected_values in attributes_to_check.items():
        if experiment_attributes[attribute] not in expected_values:
            return False
    return True


def filter_attribute(experiments_attributes, attributes_to_check):
    res = []
    for experiment_name, experiment_attribute in sorted(experiments_attributes.items()):
        matches_attributes = check_attribute(experiment_attribute, attributes_to_check)
        if matches_attributes:
            res.append(experiment_name)
    return sorted(res)


def filter_attribute_list(experiment_attributes, attributes_to_check_list):
    res = []
    for attributes_to_check in attributes_to_check_list:
        res = res + filter_attribute(
            experiments_attributes=experiment_attributes,
            attributes_to_check=attributes_to_check,
        )
    return res


def get_style(current_attributes, mapping, option_name):
    res = None
    for attribute in current_attributes:
        attr_to_consider = current_attributes[attribute]
        if attr_to_consider in mapping:
            if res is None:
                res = mapping[current_attributes[attribute]]
            else:
                raise RuntimeError(
                    f"Two options found for {option_name} and mapping {mapping}:\n",
                    f"{res} and {mapping[current_attributes[attribute]]}",
                )
    return res


def plot_evolution_iterations(
    data,
    name,
    column_name="test_acc mean",
    current_attributes=None,
    attribute_mapping=None,
):
    print(data.columns)
    data_to_consider = data[column_name].dropna()
    if current_attributes is None or attribute_mapping is None:
        plt.plot(data_to_consider.index, data_to_consider, label=name)
        return
    color = get_style(current_attributes, attribute_mapping["color"], "color")
    linestyle = get_style(
        current_attributes, attribute_mapping["linestyle"], "linestyle"
    )
    linewidth = get_style(
        current_attributes, attribute_mapping["linewidth"], "linewidth"
    )
    plt.plot(
        data_to_consider.index,
        data_to_consider,
        label=name,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )


def get_attributes_columns(name, display_attributes, data_to_plot, orderings):
    if name in display_attributes:
        column_labels = display_attributes[name]
        if isinstance(column_labels, list):
            return data_to_plot[column_labels].apply(tuple, axis=1), None
        elif isinstance(column_labels, str):
            if orderings is not None and column_labels in orderings:
                return column_labels, orderings[column_labels]
            return column_labels, None
    return None, None


def select_attributes_to_keep(display_attributes, x_axis_name) -> list[str]:
    attributes_to_keep = [] if x_axis_name == "iteration" else [x_axis_name]
    if display_attributes is not None:
        for _, item in display_attributes.items():
            if isinstance(item, list):
                for attribute in item:
                    if attribute not in attributes_to_keep:
                        attributes_to_keep.append(attribute)
            elif item not in attributes_to_keep:
                attributes_to_keep.append(item)
    return attributes_to_keep


def all_equals(l):
    if len(l) == 0:
        return True
    element = l[0]
    for i in range(1, len(l)):
        if l[i] != element:
            return False
    return True


def load_data(
    directory,
    max_machines,
    max_processes,
    machine_folder,
    result_file,
    starting_iteration,
    max_iteration,
):
    data = pd.DataFrame({})
    for machine in range(max_machines):
        for rank in range(max_processes):
            # print(f"Loading results for machine {machine} and rank {rank}.  ",end = "\r")
            uid = rank + machine * max_processes

            file = os.path.join(
                directory, machine_folder.format(machine), result_file.format(rank)
            )
            tmp_df = pd.read_json(file)
            tmp_df["uid"] = uid  # Manually add the uid for further processing
            tmp_df["iteration"] = tmp_df.index
            # print(tmp_df)
            tmp_df = tmp_df[tmp_df["iteration"] >= starting_iteration]
            tmp_df = tmp_df[tmp_df["iteration"] <= max_iteration]
            data = pd.concat([data, tmp_df])
    return data


def load_threshold_attack_results(current_experiment_data, experiment_dir):
    experiment_name = os.path.basename(experiment_dir)
    expected_file_name = "threshold_" + experiment_name + ".csv"
    directories = sorted(os.listdir(experiment_dir))

    if expected_file_name not in directories:
        print(
            f"Not loading threshold attack results: {expected_file_name} was not listed with attack results in {experiment_dir}. Entire directory:\n{directories}"
        )
        return current_experiment_data
    attacks_df = pd.read_csv(os.path.join(experiment_dir, expected_file_name))
    attacks_df = attacks_df.drop(columns="Unnamed: 0")
    attacks_df = attacks_df.rename(columns={"agent": "uid"})
    attacks_df["50auc-distance"] = np.abs(0.5 - attacks_df["roc_auc"])
    attacks_df["50auc-distance_balanced"] = np.abs(0.5 - attacks_df["roc_auc_balanced"])
    # print(attacks_df.columns)
    # print(current_experiment_data)
    # print(attacks_df)

    res = pd.merge(
        current_experiment_data, attacks_df, on=["uid", "iteration"], how="outer"
    )
    # print(res)
    return res


def load_linkability_attack_results(current_experiment_data, experiment_dir):
    experiment_name = os.path.basename(experiment_dir)
    expected_file_name = f"linkability_{experiment_name}.csv"
    directories = sorted(os.listdir(experiment_dir))
    if expected_file_name not in directories:
        print(
            f"Not loading linkability attack results: {expected_file_name} was not listed with attack results in {experiment_dir}. Entire directory:\n{directories}"
        )
        return current_experiment_data

    linkability_attack_df = pd.read_csv(
        os.path.join(experiment_dir, expected_file_name)
    )

    linkability_attack_df = linkability_attack_df.rename(columns={"agent": "uid"})

    linkability_attack_df = linkability_attack_df.drop(columns="Unnamed: 0")

    return pd.merge(
        current_experiment_data,
        linkability_attack_df,
        on=["uid", "iteration", "target"],
        how="outer",
    )


def load_biasedthreshold_results(current_experiment_data, experiment_dir):
    experiment_name = os.path.basename(experiment_dir)
    expected_file_name = f"biasedthreshold_{experiment_name}.csv"
    directories = sorted(os.listdir(experiment_dir))
    if expected_file_name not in directories:
        print(
            f"Not loading biasedthreshold attack results: {expected_file_name} was not listed with attack results in {experiment_dir}. Entire directory:\n{directories}"
        )
        return current_experiment_data

    biasedthreshold_attack_df = pd.read_csv(
        os.path.join(experiment_dir, expected_file_name)
    )

    biasedthreshold_attack_df = biasedthreshold_attack_df.rename(
        columns={"agent": "uid", "roc_auc": "biaised_roc_auc"}
    )

    biasedthreshold_attack_df = biasedthreshold_attack_df.drop(columns="Unnamed: 0")

    merged = pd.merge(
        current_experiment_data,
        biasedthreshold_attack_df,
        on=["uid", "iteration", "target"],
        how="outer",
    )

    return merged


def load_classifier_results(
    current_experiment_data, experiment_dir: os.PathLike
) -> pd.DataFrame:
    """Loads classifier attack results from an experiment file.
    Note this had to be handled separately from other attack results,
    as we cannot join on "iteration" since this attack considers multiple iterations.

    Args:
        experiment_dir (os.PathLike): The path to the experiment

    Returns:
        pd.DataFrame: The loaded CSV, with some pre-processing/micellaneous renames.
    """
    # TODO: Should I consider simply duplicating the attack results for all experiments???
    experiment_name = os.path.basename(experiment_dir)
    expected_file_name = f"classifier_{experiment_name}.csv"
    directories = sorted(os.listdir(experiment_dir))
    if expected_file_name not in directories:
        print(
            "Not loading classifier attack results: "
            + f"{expected_file_name} was not listed with attack results in {experiment_dir}."
            + f"Entire directory:\n{directories}"
        )
        return pd.DataFrame({})

    classifier_attack_df = pd.read_csv(os.path.join(experiment_dir, expected_file_name))

    classifier_attack_df = classifier_attack_df.rename(
        columns={
            "agent": "uid",
            "roc_auc": "classifier_roc_auc",
            "attacker_model": "classifier_attacker_model",
        }
    )

    classifier_attack_df = classifier_attack_df.rename(
        columns={
            f"tpr_at_fpr{fpr}": f"classifier_tpr_at_fpr{fpr}"
            for fpr in ["0.1", "0.01", "0.001", "0.0001", "1e-05"]
        }
    )
    classifier_attack_df = classifier_attack_df.drop(columns="Unnamed: 0")

    merged = pd.merge(
        current_experiment_data,
        classifier_attack_df,
        on=["uid", "target"],
        how="left",
    )

    return merged


# Legacy function to recompute some of the linkability attack results. Should only be used if there are errors in the data (see perform_attack.py).
def fix_linkability_attack_results(experiment_name, attack_results_path):
    expected_file_name = f"linkability_{experiment_name}.csv"
    directories = sorted(os.listdir(attack_results_path))
    if expected_file_name not in directories:
        print(
            f"Not loading attack results: {expected_file_name} was not listed with attack results in {attack_results_path}. Entire directory:\n{directories}"
        )
        raise FileNotFoundError(expected_file_name)

    linkability_attack_df = pd.read_csv(
        os.path.join(attack_results_path, expected_file_name)
    )

    # Fixing all the missing values/wrongly filled values
    linkability_attack_df["linkability_top1"] = (
        linkability_attack_df["linkability_top1_guess"]
        == linkability_attack_df["agent"]
    )

    linkability_attack_df.reset_index(drop=True)
    linkability_attack_df.set_index(["agent", "iteration"])
    columns = linkability_attack_df.columns.to_list()
    columns_losses = [column for column in columns if "loss_trainset_" in column]

    linkability_attack_df["linkability_real_rank"] = np.nan
    linkability_attack_df["linkability_real_rank"].astype("Int64", copy=False)
    for index, row in linkability_attack_df.iterrows():
        losses = [(int(column.split("_")[2]), row[column]) for column in columns_losses]
        losses_sorted = sorted(losses, key=lambda x: x[1])
        agents_sorted = [x[0] for x in losses_sorted]
        current_agent = row["agent"]
        linkability_rank = agents_sorted.index(current_agent)
        linkability_attack_df.at[index, "linkability_real_rank"] = linkability_rank

    linkability_attack_df = linkability_attack_df.drop(columns="Unnamed: 0")
    linkability_attack_df.to_csv(
        os.path.join(attack_results_path, f"fixed_{expected_file_name}")
    )

    return linkability_attack_df


def load_data_element(
    experiment_dir,
    starting_iteration,
    max_iteration,
    machine_folder="machine{}",
    result_file="{}_results.json",
):
    name = os.path.basename(experiment_dir)
    g5k_config_file = os.path.join(experiment_dir, "g5k_config.json")

    print(f"Loading config file {g5k_config_file}")
    with open(g5k_config_file, "r") as e:
        g5k_config = json.load(e)

    decentralizepy_config_file = os.path.join(experiment_dir, "config.ini")
    print(f"Loading config file {decentralizepy_config_file}")
    decentralizepy_config = read_ini(decentralizepy_config_file, False)

    nb_machine = g5k_config["NB_MACHINE"]
    max_processes = int(g5k_config["NB_AGENTS"] / nb_machine)

    print(f"Loading data from {name}")
    current_results = load_data(
        experiment_dir,
        max_machines=nb_machine,
        max_processes=max_processes,
        machine_folder=machine_folder,
        result_file=result_file,
        starting_iteration=starting_iteration,
        max_iteration=max_iteration,
    ).dropna()
    current_results = load_threshold_attack_results(current_results, experiment_dir)
    current_results = load_linkability_attack_results(current_results, experiment_dir)
    current_results = load_biasedthreshold_results(current_results, experiment_dir)
    current_results = load_classifier_results(current_results, experiment_dir)
    # input_dict[name] = current_results
    attributes = get_attributes(name)
    for attribute, attribute_value in attributes.items():
        current_results[attribute] = attribute_value

    current_results["seed"] = safe_load(
        decentralizepy_config, "DATASET", "random_seed", int
    )
    current_results["model"] = safe_load(
        decentralizepy_config, "DATASET", "model_class", str
    )
    current_results["lr"] = safe_load(
        decentralizepy_config, "OPTIMIZER_PARAMS", "lr", float
    )
    current_results["local_rounds"] = safe_load(
        decentralizepy_config, "TRAIN_PARAMS", "rounds", int
    )
    current_results["batch_size"] = safe_load(
        decentralizepy_config, "TRAIN_PARAMS", "batch_size", int
    )

    print(f"Finished loading data from {name}")
    return current_results


def get_experiments_dict(names, attribute_dict=ATTRIBUTE_DICT):
    experiment_dict = {}
    for filename in names:
        experiment_dict[filename] = get_attributes(filename, attribute_dict)
    return experiment_dict


def get_full_path_dict(experiments_dir):
    files_list = os.listdir(experiments_dir)
    full_path_dict = {}
    for experiment in files_list:
        full_path_dict[experiment] = os.path.join(experiments_dir, experiment)
    return full_path_dict


def format_data(
    data,
    key,
    columns_to_agg,
    linkability_aggregators,
    general_aggregator,
    total_processes,
    to_start_avg,
    noise_mapping=NOISES_MAPPING,
    noise_mapping_log=NOISE_MAPPING_LOG,
):
    usable_data = data[
        ["iteration"] + columns_to_agg + list(linkability_aggregators.keys())
    ]
    grouped_data = usable_data.groupby(["iteration"])
    usable_data = grouped_data.agg(general_aggregator)
    usable_data.reset_index(inplace=True)
    usable_data.set_index("iteration", inplace=True)

    usable_data.insert(1, "experience_name", key)
    usable_data.insert(2, "number_agents", total_processes)

    usable_data.columns = [
        " ".join(e) if len(e[-1]) > 0 else e[0] for e in usable_data.columns
    ]

    experiment_attributes = get_attributes(key)
    for attribute, attribute_value in experiment_attributes.items():
        usable_data[attribute] = attribute_value

    for attribute in ["lr", "local_rounds", "model", "seed", "batch_size"]:
        assert (
            data[attribute].nunique() == 1
        ), f"{attribute} had multiple values when it shouldn't"
        usable_data[attribute] = data[attribute].iloc[0]

    # Compute an additional column: "{column}_start_avg"
    for column_name in to_start_avg:
        rolled_average = start_avg(usable_data[column_name])
        usable_data[column_name + "_start_avg"] = rolled_average
        usable_data[column_name + "_cum_sum"] = usable_data[column_name].cumsum()
        # print(usable_data[column_name + "_start_avg"].dropna() )

    usable_data["noise_level_value"] = usable_data["noise_level"].apply(
        lambda x: noise_mapping[x]
    )
    usable_data["log_noise"] = usable_data["noise_level"].apply(
        lambda x: noise_mapping_log[x]
    )
    return usable_data


if __name__ == "__main__":
    EXPERIMENT_DIR = "attacks/my_results/movielens"
    paths_dict = get_full_path_dict(EXPERIMENT_DIR)
    experiments_dict = get_experiments_dict(paths_dict.keys())
    print(experiments_dict)
    print(paths_dict)

    attributes_list_to_check = [
        {
            "variant": "zerosum",
            "additional_attribute": ["noselfnoise"],
        },
        {
            "variant": "muffliato",
            "avgsteps": "1avgsteps",
        },
    ]

    res = filter_attribute_list(experiments_dict, attributes_list_to_check)
    print(res)
