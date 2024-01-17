import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ATTRIBUTE_DICT = {
    "network_size": ["128nodes"],
    "topology_type": ["static", "dynamic"],
    "variant": ["nonoise", "muffliato", "zerosum"],
    "avgsteps": ["10avgsteps", "1avgsteps"],
    "additional_attribute": ["selfnoise", "noselfnoise", "nonoise", "muffliato"],
    "noise_level": ["nonoise", "2th", "4th", "8th", "16th", "32th", "64th"],
}


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


def select_attributes_to_keep(display_attributes, x_axis_name):
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


def plot_all_experiments(
    data: dict[str, pd.DataFrame],
    experiments,
    display_attributes,
    plot_name,
    column_name="test_acc mean",
    save_directory=None,
    orderings=None,
):
    attributes_to_keep = select_attributes_to_keep(display_attributes, "iteration")
    data_to_plot = pd.DataFrame({})
    for experiment in experiments:
        data_experiment = data[experiment][[column_name] + attributes_to_keep]
        data_experiment = data_experiment.dropna()
        data_to_plot = pd.concat([data_to_plot, data_experiment])

    sns.set_theme()
    hue, hue_ordering = get_attributes_columns(
        "hue", display_attributes, data_to_plot, orderings
    )
    style, style_ordering = get_attributes_columns(
        "style", display_attributes, data_to_plot, orderings
    )
    size, size_ordering = get_attributes_columns(
        "size", display_attributes, data_to_plot, orderings
    )
    col, col_ordering = get_attributes_columns(
        "col", display_attributes, data_to_plot, orderings
    )

    plot = sns.relplot(
        data=data_to_plot,
        kind="line",
        x="iteration",
        y=column_name,
        hue=hue,
        hue_order=hue_ordering,
        style=style,
        style_order=style_ordering,
        size=size,
        size_order=size_ordering,
        col=col,
        col_order=col_ordering,
        sizes=(1, 5),
    )
    plot.fig.suptitle(plot_name)
    plot.fig.subplots_adjust(top=0.9)
    if save_directory is not None:
        savefile = f"{save_directory}{plot_name.replace(' ','_')}.pdf"
        print(f"Saving to {savefile}")
        plt.savefig(savefile)
    return


def scatter_all_experiments(
    data: dict[str, pd.DataFrame],
    experiments,
    display_attributes,
    plot_name,
    x_axis_name="iteration",
    column_name="test_acc mean",
    save_directory=None,
    orderings=None,
):
    attributes_to_keep = select_attributes_to_keep(
        display_attributes=display_attributes, x_axis_name=x_axis_name
    )
    data_to_plot = pd.DataFrame({})
    for experiment in experiments:
        data_experiment = data[experiment][[column_name] + attributes_to_keep]
        data_experiment = data_experiment.dropna()
        data_to_plot = pd.concat([data_to_plot, data_experiment])

    sns.set_theme()
    hue, hue_ordering = get_attributes_columns(
        "hue", display_attributes, data_to_plot, orderings
    )
    style, style_ordering = get_attributes_columns(
        "style", display_attributes, data_to_plot, orderings
    )
    size, size_ordering = get_attributes_columns(
        "size", display_attributes, data_to_plot, orderings
    )
    col, col_ordering = get_attributes_columns(
        "col", display_attributes, data_to_plot, orderings
    )

    plot = sns.jointplot(
        data=data_to_plot,
        x=x_axis_name,
        y=column_name,
        hue=hue,
        hue_order=hue_ordering,
        # style=style,
        # style_order=style_ordering,
        # size=size,
        # size_order=size_ordering,
        # col=col,
        # col_order=col_ordering,
        # sizes=(5, 10),
    )
    plot.fig.suptitle(plot_name)
    plot.fig.subplots_adjust(top=0.9)
    if save_directory is not None:
        savefile = f"{save_directory}{plot_name.replace(' ','_')}.pdf"
        print(f"Saving to {savefile}")
        plt.savefig(savefile)
    return


def scatter_aggregated_data(
    data,
    name,
    column_name="test_acc mean",
    x_axis_name="iteration",
    current_attributes=None,
    attribute_mapping=None,
    method="mean",
):
    if x_axis_name == "iteration":
        raise ValueError("Shouldn't scatter averaged data on iterations")
    else:
        data_to_consider = data[[x_axis_name, column_name]].dropna()
        if method == "mean":
            x_value = data_to_consider[x_axis_name].mean()
            y_value = data_to_consider[column_name].mean()
        elif method == "max":
            x_value = data_to_consider[x_axis_name].max()
            y_value = data_to_consider[column_name].max()
        else:
            raise ValueError(f"Expected method in ['mean', 'max'], got {method}")
    if current_attributes is None or attribute_mapping is None:
        plt.scatter(x_value, y_value, label=name)
        return
    color = get_style(current_attributes, attribute_mapping["color"], "color")
    marker_style = get_style(current_attributes, attribute_mapping["marker"], "marker")
    linewidth = get_style(
        current_attributes, attribute_mapping["linewidth"], "linewidth"
    )
    plt.scatter(
        x=x_value,
        y=y_value,
        label=name,
        color=color,
        marker=marker_style,
        linewidth=linewidth,
    )


def scatter_averaged_experiments(
    data,
    experiments,
    experiments_attributes,
    display_attributes,
    plot_name,
    column_name="test_acc mean",
    x_axis_name="iteration",
    figsize=(25, 25),
    save_directory=None,
    method="mean",
):
    plt.figure(figsize=figsize)
    for experiment in sorted(experiments):
        scatter_aggregated_data(
            data=data[experiment],
            name=experiment,
            column_name=column_name,
            x_axis_name=x_axis_name,
            current_attributes=experiments_attributes[experiment],
            attribute_mapping=display_attributes,
            method=method,
        )

    plt.legend()
    plt.ylabel(f"{column_name} {method}")
    plt.xlabel(f"{x_axis_name} {method}")
    plt.title(plot_name)
    if save_directory is not None:
        savefile = f"{save_directory}{plot_name.replace(' ','_')}.pdf"
        print(f"Saving to {savefile}")
        plt.savefig(savefile)
    return


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
        linkability_rank = agents_sorted.index(row["agent"])
        linkability_attack_df.at[index, "linkability_real_rank"] = linkability_rank

    linkability_attack_df = linkability_attack_df.drop(columns="Unnamed: 0")
    linkability_attack_df.to_csv(
        os.path.join(attack_results_path, f"fixed_{expected_file_name}")
    )

    return linkability_attack_df


if __name__ == "__main__":
    EXPERIMENT_DIR = "results/my_results/icml_experiments/cifar10"
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
