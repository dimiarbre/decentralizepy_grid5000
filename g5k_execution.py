import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import enoslib as en

from utils import *

args = sys.argv

IS_REMOTE = False
if len(args) < 2:
    G5K_CONFIG_PATH = "small_test_run.json"
    print(
        f'Using default config "{G5K_CONFIG_PATH}" as no arg (or too much args) was provided.'
        + "Setting to debug mode"
    )
    G5K_CONFIG_PATH = "g5k_config/" + G5K_CONFIG_PATH
    DEBUG = True
elif len(args) == 2:
    G5K_CONFIG_PATH = args[1]
    DEBUG = False

else:
    G5K_CONFIG_PATH = args[1]
    if args[2] in ["DEBUG", "Debug", "debug"]:
        DEBUG = True
    elif args[2] in ["Remote", "remote", "REMOTE"]:
        IS_REMOTE = True
        DEBUG = False
    else:
        DEBUG = False

print(f"Debugging is set to {DEBUG}!")


with open(G5K_CONFIG_PATH, "r") as g5k_config_file:
    g5k_config = json.load(g5k_config_file)

print(json.dumps(g5k_config, indent=8))

job_name = g5k_config["job_name"]
walltime = g5k_config["walltime"]
GRAPH_FILE = g5k_config["GRAPH_FILE"]
NB_AGENTS = g5k_config["NB_AGENTS"]  # Must be divisible by NB_MACHINE!
NB_MACHINE = g5k_config["NB_MACHINE"]
NB_ITERATION = g5k_config["NB_ITERATION"]
EVAL_FILE = g5k_config["EVAL_FILE"]
TEST_AFTER = g5k_config["TEST_AFTER"]
LOG_LEVEL = g5k_config["LOG_LEVEL"]
CONFIG_NAME = g5k_config["CONFIG_NAME"]
cluster = g5k_config["cluster"]
LOCAL_SAVE_DIR = g5k_config["LOCAL_SAVE_DIR"]
if "AVERAGING_STEPS" in g5k_config.keys():
    AVERAGING_STEPS = g5k_config["AVERAGING_STEPS"]
else:
    AVERAGING_STEPS = 1
if "DOWNLOAD_ALL" in g5k_config.keys():
    DOWNLOAD_ALL = g5k_config["DOWNLOAD_ALL"] == "True"
else:
    DOWNLOAD_ALL = False


extra_time = "00:10:00"
real_walltime = add_times(walltime, extra_time)

# TODO: automatically detect the closest we can be, and give less agents to the last machine.
assert (
    NB_AGENTS % NB_MACHINE == 0
), f"Invalid number of agent {NB_AGENTS} for the number of machines {NB_MACHINE}, as the correct number is not computed dynamically"
NB_PROC_PER_MACHINE = NB_AGENTS // NB_MACHINE
CONFIG_FILE = "run_configuration/" + CONFIG_NAME


with open(CONFIG_FILE) as config:
    config_content = config.readlines()
    config_content = "".join(config_content)
    print(config_content)

en.init_logging(level=logging.INFO)
en.check()

# Set very high parallelism to be able to handle a large number of VMs
# print(NB_MACHINE)
en.set_config(ansible_forks=NB_MACHINE)  # Only works with enoslib>=9.0.0


conf = (
    # en.G5kConf.from_settings(job_name=job_name, walltime=real_walltime, job_type=["exotic"])
    en.G5kConf.from_settings(job_name=job_name, walltime=real_walltime)
    .add_machine(roles=["head"], cluster=cluster, nodes=1)
    .add_machine(
        roles=["agent"],
        cluster=cluster,
        nodes=NB_MACHINE - 1,
    )
)

provider = en.G5k(conf)
roles, networks = provider.init()

cpu_info = en.run_command("lscpu; cat /proc/cpuinfo", roles=roles)

# Install glances for profiling
result = en.run_command("sudo-g5k apt install -y glances", roles=roles)

result = en.run_command("echo $OAR_JOB_ID", roles=roles["head"])
job_id = result[0].stdout
print(f"Job ID : {job_id}")


SINGULARITY = "/grid5000/spack/v1/opt/spack/linux-debian11-x86_64_v2/gcc-10.4.0/singularity-3.8.7-rv6m5rw2bda5vu5cb7kcw6jfjg24xp6h/bin/singularity"
REMOTE_SCRATCH_DIR = "/home/dlereverend/scratch"
REMOTE_INPUT_DIR = REMOTE_SCRATCH_DIR + "/decentralizepy_grid5000"
REMOTE_DECENTRALIZEPY_DIR = REMOTE_INPUT_DIR + "/decentralizepy"
REMOTE_LOGS_DIR = "/tmp/logs"
REMOTE_DATASET_DIR = "/home/dlereverend/datasets"
CONTAINER_FILE = REMOTE_INPUT_DIR + "/compute_container.sif"
RUN_FOLDER_NAME = f"{job_id}_{job_name}"
REMOTE_RESULT_DIR = REMOTE_SCRATCH_DIR + f"/results/{RUN_FOLDER_NAME}"


singularity_version = en.run_command(f"{SINGULARITY} --version", roles=roles)

REMOTE_PRIVACY_DATASETS_DIR = "/tmp/privacy"
REMOTE_IP_FILE = REMOTE_LOGS_DIR + "/ip.json"

print("--------Creating all the necessary directories--------")
with en.actions(roles=roles) as a:
    a.file(path=REMOTE_LOGS_DIR, state="directory")
    a.file(path=REMOTE_PRIVACY_DATASETS_DIR, state="directory")
    a.file(path=REMOTE_DATASET_DIR, state="directory")

# Setup the config file.
result = en.run_command(
    f'echo "{config_content}">{REMOTE_LOGS_DIR}/config.ini', roles=roles
)

# result = en.run_command(f"{SINGULARITY} --help", roles=roles)

# roles = en.sync_info(roles, networks)

print("--------Creating IP file---------")

hostname_results = en.run_command("hostname -I| awk '{print $1;}'", roles=roles)

ip_file_content = "{"

for id_machine, result in enumerate(hostname_results):
    ip_addr = result.stdout
    if id_machine == 0:
        ip_file_content += f'\n\t"{id_machine}":"{ip_addr}"'
    else:
        ip_file_content += f',\n\t"{id_machine}":"{ip_addr}"'

# h = roles["head"][0]
# for j,addr in enumerate(h.filter_addresses(networks["prod"])): #To deal with multiple ip addresses in a single node?
#     ip_addr =  addr.ip.ip
#     print(f"Got ip address {addr.ip}")
#     if j==0: # The first line:


# id_machine +=1

# for h in roles["agent"]:
#     for j,addr in enumerate(h.filter_addresses(networks["prod"])): #To deal with multiple ip addresses in a
#         ip_addr =  addr.ip.ip
#         ip_file_content += f",\n\t\"{id_machine}\":\"{ip_addr}\""
#     id_machine +=1

ip_file_content += "\n}"
print(ip_file_content)

# TODO: change to accomodate for several jobs?
result = en.run_command(f"echo '{ip_file_content}'>{REMOTE_IP_FILE}", roles=roles)

# Launching job on each machine:
text_command = (
    f"{SINGULARITY} run --bind {REMOTE_DATASET_DIR}:/datasets "
    + f"--bind {REMOTE_DECENTRALIZEPY_DIR}:/decentralizepy "
    + f"--bind {REMOTE_LOGS_DIR}:/logs "
    + f"--bind {REMOTE_IP_FILE}:/ip.json "
    + f"--bind {REMOTE_PRIVACY_DATASETS_DIR}:/privacy_datasets "
    + f"{CONTAINER_FILE} {GRAPH_FILE} {NB_MACHINE} {NB_PROC_PER_MACHINE} {NB_ITERATION} {EVAL_FILE} {TEST_AFTER} {LOG_LEVEL} {AVERAGING_STEPS}"
)


def save_results(download_logs=False):
    # Saves the results on the g5k global storage
    print("-" * 20 + "saving results, discarding logs if needed" + "-" * 20)
    with en.actions(roles=roles) as a:
        a.file(path=REMOTE_RESULT_DIR, state="directory")

    # Backup the logs to main g5k storage
    result = en.run_command(
        f"rsync -Crvz {REMOTE_LOGS_DIR}/* {REMOTE_RESULT_DIR}/", roles=roles["head"]
    )
    # Conditional download of logs to ease the load on the remote storage
    if download_logs:
        synchro_command = f'rsync -Crvz --exclude "ip.json" --exclude "*.ini"  {REMOTE_LOGS_DIR}/* {REMOTE_RESULT_DIR}/'
    else:
        synchro_command = f'rsync -Crvz --exclude "ip.json"  --exclude "*.log" --exclude "*.ini"  {REMOTE_LOGS_DIR}/* {REMOTE_RESULT_DIR}/'

    result = en.run_command(
        synchro_command,
        roles=roles["agent"],
    )

    # Copy and save the g5k config to the logs:
    result = en.run_command(
        f"echo '{json.dumps(g5k_config,indent=8)}'>{REMOTE_RESULT_DIR}/g5k_config.json",
        roles=roles["head"][0],
    )


def download_results(download_logs=False):
    print(f"{'-'*20}Downloading main results{'-'*20}")
    # Download the results locally
    print(f"Downloading at {LOCAL_SAVE_DIR}")
    # TODO : This only works for clusters in Rennes, check how to generalize to other clusters?
    result_main_download = subprocess.run(
        [
            "rsync",
            "-avzP",
            "--exclude",
            "**.log",
            "--exclude",
            "**.png",
            f"rennes.g5k:{REMOTE_RESULT_DIR}",  # TODO: change this to run on another site
            f"{LOCAL_SAVE_DIR}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    print(result_main_download.stderr)
    print(result_main_download.returncode)

    # Also download the logs. We want to download the results first in case of a problem
    if download_logs:
        print(f"{'-'*20}Downloading additional logs{'-'*20}")

        result_logs_download = subprocess.run(
            [
                "rsync",
                "-avzP",
                f"rennes.g5k:{REMOTE_RESULT_DIR}",  # TODO: change this to run on another site
                f"{LOCAL_SAVE_DIR}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        print(result_logs_download.returncode)
        print(result_logs_download.stderr)
        return (result_main_download.returncode == 0) and (
            result_logs_download.returncode == 0
        )
    # Return True if the download was a success
    return result_main_download.returncode == 0


def clear_results():
    """Remove the results from the g5k storage, be sure to call "download_results" before this"""
    print("Clearing remote logs")
    # Obtain a small job
    result_rm = subprocess.run(
        [
            "ssh",
            "rennes.g5k",
            f"`rm {REMOTE_RESULT_DIR} -rf`",
        ]
    )
    print(result_rm)
    return


print(f"Current job ID : {job_id}")
print(f"Debugging to {DEBUG}, NB_MACHINE to {NB_MACHINE}")
if DEBUG and NB_MACHINE == 1:
    # When in debug mode, simply print the command to run and do not free any ressource
    # This will allow to launch directly from the machine and thus have stdout access
    # TODO: launch on every machine but one?
    print("sudo-g5k rm -r /tmp/logs/machine*; time sudo-g5k " + text_command)
else:
    # Run  the main command automatically any other case.
    target_walltime_sec = to_sec(walltime)
    print(f"Enoslib version: {en.__version__}")
    t0 = time.time()
    try:
        main_result = en.run_command(
            text_command, roles=roles, asynch=target_walltime_sec, poll=5 * 60
        )
    except en.errors.EnosFailedHostsError as e:
        print(e)
        raise RuntimeError from e
    t1 = time.time()

    save_results(download_logs=DOWNLOAD_ALL)

    # Free the ressource
    provider.destroy()

    if not IS_REMOTE:
        # Download the logs locally only if not running on a remote job
        download_success = download_results(download_logs=DOWNLOAD_ALL)

        if download_success:
            # Successfully pulled all the files, then delete everything to clear space on g5k
            clear_results()
        else:
            print(
                "WARNING: Download seems to have failed. Data not cleared, check manually"
            )
    print(
        f"Job finished normally and was deleted, main command took {(t1-t0)/(60*60):.2f} hours to run."
    )
