# Decentralizepy & Grid5000
This git will represent an adaptation of the decentralizepy framework to run on Grid5000. The main framework is the one of the [EPFL](https://github.com/sacs-epfl/decentralizepy), see [Decentralized Learning Made Easy with DecentralizePy](https://arxiv.org/abs/2304.08322).

This project is designed to run on multiple servers, using potentially multiple machines, as a single machine may not have enough resources to run the experiment. 


## Running an experiment

This git is designed to run for a singularity container (using the `compute_container.def` file). To build the container, run :
```bash
singularity build --fakeroot compute_container.sif compute_container.def
```
Or, on G5k:

```bash
sudo-g5k /grid5000/spack/v1/opt/spack/linux-debian11-x86_64_v2/gcc-10.4.0/singularity-3.8.7-rv6m5rw2bda5vu5cb7kcw6jfjg24xp6h/bin/singularity -d build compute_container.sif compute_container.def
```

Alternatively, you can now run 
```bash
make
```

This should install all the dependencies in the container file, `compute_container.sif`. 

If needed, this command should be run on all of your remote servers if their storage is not synced.

This container expects several arguments to run. 
```
singularity run compute_container.sif 
graph : The graph file, located in src/decentralizepy/run_configuration

nb_machines : the number of machine the process will run on 

nb_procs_per_machine : how many process this machine will simulate

iterations : Number of communication rounds, be careful not to confuse with `rounds` the number of local gradient descent 
    
eval_file: The file that will be used to run the simulation. Either "testingPeerSampler.py" or "testingPeerSamplerDynamic.py", to distinguish between a static or dynamic network.

test_after: Every test_after iterations, the entire train loss is logged

log_level: Either "INFO" or "DEBUG"

averaging_steps: The number or averaging rounds, use 1 for ZeroSum, and 10 was used in the paper.
```

The singularity file also expects specific file mounting to be able to modify some files. 

### How to mount files

#### Using Grid5000
If you can use Grid5000, then all the running process and logs handling is automated using `g5k_execution.py`. You simply need to edit g5k configuration file in `g5k_config/`, and run
```
python3 g5k_execution.py g5k_config/my_config.json
```
This grid5000 configuration expects a correct decentralizepy configuration file (see the following sections)

To run, you need several dependencies, and in particular the library [enoslib](https://discovery.gitlabpages.inria.fr/enoslib/tutorials/grid5000.html): 
```
pip3 install enoslib
```

This script will setup all the necessary files and mount them in the singularity call automatically. Simply follow one of the examples, such as `g5k_config/mia_config_cifar10_zerosum.json`, and you can create your own configuration and run it. It is important to change the name for reservations not to overlap when running multiple experiments at a time.

For now, it is mostly designed and tuned to run on the `paravance` node (logs downloading and timings). But this can be quickly fixed in the code itself for other nodes.

#### On another architecture:
You need to be able to run all the singularity files on all the networks. Then, the files in the following section will need to be created manually and made available on all the machines. Here is a list, and where they are expected to be mounted using the option `--bind`:
```
--bind $(REMOTE_DECENTRALIZEPY_DIR):/decentralizepy
--bind $(REMOTE_LOGS_DIR):/logs
--bind $(REMOTE_IP_FILE):/ip.json
--bind {REMOTE_DATASET_DIR}:/datasets
--bind $(REMOTE_PRIVACY_DATASETS_DIR):/privacy_datasets 
```
Here, each variable must be set up appropriately according to your architecture.


### The decentralizepy configuration file
Then, the hyperparameters for the experiments must be set. 
To create a run configuration, go to `decentralizepy/run_configuration`, and set up the configuration that will be used. For the paper, use 
* For CIFAR10: `mia_config_cifar10_muffliato.ini` or `mia_config_cifar10_zerosum.ini`  
* For FEMNIST: `mia_config_femnist_muffliato.ini` or `mia_config_femnist_zerosum.ini`
You simply need to go to the file and specify the noise level you want for the run.

This file should be located in `decentralizepy/run_configuration`, and since decentralizepy will be mounted then the script will access this file directly.  


An important note, depending on the architecture used, there can be a TCP port conflict, with a TCP port being already used. For this case, an additional parameter can be added in the [COMMUNICATION] section if there is a TCP port conflict. The default is the following, but can be changed in the config file:
```
offset=9000
```


### The logs directory:
This directory is the one where all the logs will be written, as well as the results. This directory should then be synced across all the machines for the data-extraction script to work properly.


### The IP file
This file is the most important: it must be a json format file that creates an association `machine_ID : ip_address`, listing all the IP addresses of the machines that will partake in the training. This will allow the processes to communicate. The container will then identify automatically the machine ID.
It must be the same file for all  

### The dataset folder
You must have the dataset downloaded on all the machines, and it will the dataset automatically. In the case of CIFAR10, it can also be downloaded automatically. 

### The privacy dataset folder
Under the current configurations, this dataset should not be used. You can simply set any folder in this.


## Using the results
This section will all be based in the `results/` folder.

Once you downloaded the results, you can use `data_refactoring_128nodes.ipynb` notebook: rename all the experiments paths appropriately in the first cell, and the script will generate CSV files that gathers all the results.

Once this data is refactored, you can additionally perform preliminary plots using `privacyplots_refactoreddata.ipynb`: set up the correct folder containing the refactored results, and for each plot select which elements need to be plotted. It will also save each figure as a pdf file.  

## Todo 
* [x] Test building
* [x] Add another file that can run on local machines.
* [x] Implement privacy attacks to reproduce previous results
* [x] Adapt to Grid'5000
* [x] Make the script adapt to add the offset directly into the communication section (for .ini files)
* [ ] GPU support
* [ ] Remove `data_refactoring.ipynb` and rename `data_refactoring_128nodes.ipynb`


## Known issues:
* Launching with `NB_PROC_PER_MACHINE>1` and no dataset already downloaded and formatted will cause a crash, as multiple processes will try to download it. Launch a fake job with `NB_ITERATION=0` and  `NB_PROC_PER_MACHINE=1` beforehand to fix the problem, and have the dataset be on a shared drive so that it only needs to be downloaded once.
* GPU support is not fully functional: there is a bug when considering MIA attacks with GPU. No quick fix could be found, a solution is to switch it all to cpu before changing anything. For these reasons, GPU support is deactivated (no `--nv` option in the singularity run call) even though there are code fragments for GPU support.