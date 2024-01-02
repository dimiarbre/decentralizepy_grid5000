import os
import matplotlib.pyplot as plt

ATTRIBUTE_DICT = {
    "network_size": ["128nodes"],
    "topology_type": ["static", "dynamic"],
    "variant":["nonoise","muffliato","zerosum"],
    "avgsteps": ["10avgsteps"],
    "additional_attribute": ["selfnoise","noselfnoise"],
    "noise_level":["2th","4th","8th","16th","32th","64th"],
}




def get_attributes(filename,attribute_dict = ATTRIBUTE_DICT):
    parsed_filename = filename.split("_")
    res_attribute = {}
    for global_attribute,possible_values in attribute_dict.items():
        for current_attribute in parsed_filename :
            if current_attribute in possible_values:
                if current_attribute not in res_attribute:
                    res_attribute[global_attribute] = current_attribute
                else:
                    raise RuntimeError(f"Duplicate of attributes in {filename}")

        if global_attribute not in res_attribute:
            # print(f"No value found for attribute {global_attribute} for {filename}")
            res_attribute[global_attribute] = None
    return res_attribute

def get_experiments_dict(names,attribute_dict = ATTRIBUTE_DICT):
    experiment_dict = {}
    for filename in names:
        experiment_dict[filename] = get_attributes(filename,attribute_dict)
    return experiment_dict


def get_full_path_dict(experiments_dir):
    files_list = os.listdir(experiments_dir)
    full_path_dict = {}
    for experiment in files_list:
        full_path_dict[experiment] = os.path.join(experiments_dir,experiment)
    return full_path_dict

def check_attribute(experiment_attributes,attributes_to_check):
    for attribute,expected_values in attributes_to_check.items():
        if experiment_attributes[attribute] not in expected_values:
            return False
    return True

def filter_attribute(experiments_attributes, attributes_to_check):
    res = []
    for experiment_name, experiment_attribute in experiments_attributes.items():
        matches_attributes = check_attribute(experiment_attribute, attributes_to_check)
        if matches_attributes:
            res.append(experiment_name)
    return res

def get_style(current_attributes,mapping,option_name):
    res = None
    for attribute in current_attributes:
        attr_to_consider =  current_attributes[attribute]
        if attr_to_consider in mapping :
            if res is None:
                res = mapping[current_attributes[attribute]]
            else:
                raise RuntimeError(
                    f"Two options found for {option_name} and mapping {mapping}:\n",
                    f"{res} and {mapping[current_attributes[attribute]]}"
                )
    return res

def plot_accuracy(data,name,current_attributes = None,attribute_mapping = None):
    if current_attributes is None or attribute_mapping is None:
        plt.plot(data.index,data["test_acc mean"],label=name)
        return
    color = get_style(current_attributes,attribute_mapping["color"],"color")
    linestyle = get_style(current_attributes,attribute_mapping["linestyle"],"linestyle")
    linewidth = get_style(current_attributes,attribute_mapping["linewidth"],"linewidth")
    plt.plot(data.index,data["test_acc mean"], label = name, color = color, linestyle=linestyle,linewidth=linewidth) 

if __name__=="__main__":
    EXPERIMENT_DIR = "results/my_results/icml_experiments/cifar10"
    paths_dict = get_full_path_dict(EXPERIMENT_DIR)
    experiments_dict = get_experiments_dict(paths_dict.keys())
    # print(experiments_dict)
    # print(paths_dict)

    attributes_to_check = {"variant":"zerosum","additional_attribute": ["noselfnoise"]}

    res = filter_attribute(experiments_dict, attributes_to_check)
    print(res)
