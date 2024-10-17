import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import enoslib as en

from utils import add_times, generate_config, generate_config_files, to_sec


def save_results(
    remote_result_dir, remote_logs_dir, roles, g5k_config, download_logs=False
):
    # Saves the results on the g5k global storage
    print("-" * 20 + "saving results, discarding logs if needed" + "-" * 20)
    if download_logs:
        print("SYNC OF ALL LOGS")
    with en.actions(roles=roles) as a:
        a.file(path=remote_result_dir, state="directory")

    # Backup the logs to main g5k storage
    # Conditional download of logs to ease the load on the remote storage
    if not download_logs:
        command = f"rsync -Crvz --exclude '**/graphs/*' --exclude '*.png' --exclude '*.log' {remote_logs_dir}/* {remote_result_dir}/"
        # Do not back up the graphs unless absolutely necessary (they are here for debugging purposes)
        result = en.run_command(command, roles=roles["head"])

        # Save only one log file just for safety.
        command = (
            f"rsync -Crvz {remote_logs_dir}/machine*/0.log {remote_result_dir}/0.log"
        )
        result = en.run_command(command, roles=roles["head"])
        synchro_command = f'rsync -Crvz --exclude "**/graphs/*" --exclude "ip.json" --exclude "*.ini" --exclude "*.log"  {remote_logs_dir}/* {remote_result_dir}/'
    else:
        result = en.run_command(
            f"rsync -Crvz {remote_logs_dir}/* {remote_result_dir}/", roles=roles["head"]
        )
        synchro_command = f'rsync -Crvz --exclude "ip.json" --exclude "*.ini"  {remote_logs_dir}/* {remote_result_dir}/'

    result = en.run_command(
        synchro_command,
        roles=roles["agent"],
    )

    # Copy and save the g5k config to the logs:
    result = en.run_command(
        f"echo '{json.dumps(g5k_config,indent=8)}'>{remote_result_dir}/g5k_config.json",
        roles=roles["head"][0],
    )


def download_results(local_save_dir, remote_result_dir, download_logs=False):
    print(f"{'-'*20}Downloading main results{'-'*20}")
    # Download the results locally
    print(f"Downloading at {local_save_dir}")
    # TODO : This only works for clusters in Rennes, check how to generalize to other clusters?
    result_main_download = subprocess.run(
        [
            "rsync",
            "-avzP",
            "--exclude",
            "**.log",
            "--exclude",
            "**.png",
            f"rennes.g5k:{remote_result_dir}",  # TODO: change this to run on another site
            f"{local_save_dir}",
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
                f"rennes.g5k:{remote_result_dir}",  # TODO: change this to run on another site
                f"{local_save_dir}",
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


def clear_results(remote_result_dir):
    """Remove the results from the g5k storage, be sure to call "download_results" before this"""
    print("Clearing remote logs")
    # Obtain a small job
    result_rm = subprocess.run(
        [
            "ssh",
            "rennes.g5k",
            f"`rm {remote_result_dir} -rf`",
        ]
    )
    print(result_rm)
    return


SINGULARITY = "/grid5000/spack/v1/opt/spack/linux-debian11-x86_64_v2/gcc-10.4.0/singularity-3.8.7-rv6m5rw2bda5vu5cb7kcw6jfjg24xp6h/bin/singularity"
REMOTE_SCRATCH_DIR = "/home/dlereverend/scratch"
REMOTE_INPUT_DIR = REMOTE_SCRATCH_DIR + "/decentralizepy_grid5000"
REMOTE_DECENTRALIZEPY_DIR = REMOTE_INPUT_DIR + "/decentralizepy"
REMOTE_LOGS_DIR = "/tmp/logs"
REMOTE_DATASET_DIR = "/home/dlereverend/datasets"
CONTAINER_FILE = REMOTE_INPUT_DIR + "/compute_container.sif"
REMOTE_GROUP_STORAGE = "/home/dlereverend/wide_storage/dlereverend_zerosum"


def launch_experiment(g5k_config, decentralizepy_config: str, debug, is_remote):
    true_job_name = g5k_config["job_name"]
    # Job names over 100 characters get a bad querry error.
    if len(true_job_name) > 100:
        job_name = str(hash(true_job_name))
        job_name = job_name[
            : min(100, len(job_name))
        ]  # Only keep the first hundred characters to have a valid job name
        print(f"Detected name too long {true_job_name}.\nHashed job name:{job_name}")
    else:
        job_name = true_job_name
    walltime = g5k_config["walltime"]
    graph_file = g5k_config["GRAPH_FILE"]
    nb_agents = g5k_config["NB_AGENTS"]  # Must be divisible by NB_MACHINE!
    nb_machine = g5k_config["NB_MACHINE"]
    nb_iteration = g5k_config["NB_ITERATION"]
    eval_file = g5k_config["EVAL_FILE"]
    test_after = g5k_config["TEST_AFTER"]
    log_level = g5k_config["LOG_LEVEL"]
    cluster = g5k_config["cluster"]
    local_save_dir = g5k_config["LOCAL_SAVE_DIR"]
    if "AVERAGING_STEPS" in g5k_config.keys():
        averaging_steps = g5k_config["AVERAGING_STEPS"]
    else:
        averaging_steps = 1
    if "DOWNLOAD_ALL" in g5k_config.keys():
        download_all = g5k_config["DOWNLOAD_ALL"] == "True"
    else:
        download_all = False
    if "queue" in g5k_config.keys():
        queue = g5k_config["queue"]
    else:
        queue = "default"

    if "job_type" in g5k_config:
        # Disregards night constraints if the queue is production.
        if g5k_config["job_type"] == [[]] or queue == "production":
            job_type = []
        else:
            job_type = g5k_config["job_type"]
    else:
        job_type = []
    storage_location = "HOME"
    if "STORAGE" in g5k_config:
        storage_location = "GROUP"

    extra_time = "00:10:00"
    real_walltime = add_times(walltime, extra_time)

    # TODO: automatically detect the closest we can be, and give less agents to the last machine.
    assert (
        nb_agents % nb_machine == 0
    ), f"Invalid number of agent {nb_agents} for the number of machines {nb_machine}, as the correct number is not computed dynamically"
    nb_proc_per_machine = nb_agents // nb_machine

    en.init_logging(level=logging.INFO)
    en.check()

    # Set very high parallelism to be able to handle a large number of VMs
    # print(NB_MACHINE)
    en.set_config(ansible_forks=nb_machine)  # Only works with enoslib>=9.0.0

    conf = (
        # en.G5kConf.from_settings(job_name=job_name, walltime=real_walltime, job_type=["exotic"])
        en.G5kConf.from_settings(
            job_name=job_name, walltime=real_walltime, queue=queue, job_type=job_type
        )
        .add_machine(roles=["head"], cluster=cluster, nodes=1)
        .add_machine(
            roles=["agent"],
            cluster=cluster,
            nodes=nb_machine - 1,
        )
    )
    provider = en.G5k(conf)
    roles, networks = provider.init()

    time_start = time.time()

    cpu_info = en.run_command("lscpu; cat /proc/cpuinfo", roles=roles)

    # Install glances for profiling
    # result = en.run_command("sudo-g5k apt install -y glances", roles=roles)

    result = en.run_command("echo $OAR_JOB_ID", roles=roles["head"])
    job_id = result[0].stdout
    print(f"Job ID : {job_id}")

    run_folder_name = f"{job_id}_{true_job_name}"
    # We need the job id for this section to work, thus we must alreay have the reservation
    if storage_location == "GROUP":
        remote_result_dir = os.path.join(
            REMOTE_GROUP_STORAGE, f"results/{run_folder_name}"
        )
    else:
        remote_result_dir = os.path.join(
            REMOTE_SCRATCH_DIR, f"results/{run_folder_name}"
        )
    print(remote_result_dir)
    singularity_version = en.run_command(f"{SINGULARITY} --version", roles=roles)

    remote_ip_file = REMOTE_LOGS_DIR + "/ip.json"

    print("--------Creating all the necessary directories--------")
    with en.actions(roles=roles) as a:
        a.file(path=REMOTE_LOGS_DIR, state="directory")
        a.file(path=REMOTE_DATASET_DIR, state="directory")

    # Setup the config file.
    result = en.run_command(
        f'echo "{decentralizepy_config}">{REMOTE_LOGS_DIR}/config.ini', roles=roles
    )

    # result = en.run_command(f"{SINGULARITY} --help", roles=roles)

    # roles = en.sync_info(roles, networks)

    print("--------Creating IP file---------")

    hostname_results = en.run_command("hostname -I| awk '{print $1;}'", roles=roles)

    # TODO: create a json, and convert to string afterwards,
    # instead of this janky json file generation.
    ip_file_content = "{"

    for id_machine, result in enumerate(hostname_results):
        ip_addr = result.stdout
        if id_machine == 0:
            ip_file_content += f'\n\t"{id_machine}":"{ip_addr}"'
        else:
            ip_file_content += f',\n\t"{id_machine}":"{ip_addr}"'

    ip_file_content += "\n}"
    print(ip_file_content)

    # TODO: change to accomodate for several jobs?
    result = en.run_command(f"echo '{ip_file_content}'>{remote_ip_file}", roles=roles)

    # Launching job on each machine:
    text_command = (
        f"{SINGULARITY} run --bind {REMOTE_DATASET_DIR}:/datasets "
        + f"--bind {REMOTE_DECENTRALIZEPY_DIR}:/decentralizepy "
        + f"--bind {REMOTE_LOGS_DIR}:/logs "
        + f"--bind {remote_ip_file}:/ip.json "
        + f"{CONTAINER_FILE} {graph_file} {nb_machine} {nb_proc_per_machine} {nb_iteration} {eval_file} {test_after} {log_level} {averaging_steps}"
    )

    print(f"Current job ID : {job_id}")
    print(f"Debugging to {debug}, NB_MACHINE to {nb_machine}")
    if debug and nb_machine == 1:
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

        save_results(
            g5k_config=g5k_config,
            remote_result_dir=remote_result_dir,
            remote_logs_dir=REMOTE_LOGS_DIR,
            roles=roles,
            download_logs=download_all,
        )

        # Free the ressource
        provider.destroy()

        if not is_remote:
            # Download the logs locally only if not running on a remote job
            download_success = download_results(
                local_save_dir=local_save_dir,
                remote_result_dir=remote_result_dir,
                download_logs=download_all,
            )

            if download_success:
                # Successfully pulled all the files, then delete everything to clear space on g5k
                clear_results(remote_result_dir=remote_result_dir)
            else:
                print(
                    "WARNING: Download seems to have failed. Data not cleared, check manually"
                )
        print(
            f"Job finished normally and was deleted, main command took {(t1-t0)/(60*60):.2f} hours to run."
        )
    time_finish = time.time()
    duration = time_finish - time_start
    return provider, duration


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "g5k_config",
        help="Path to the g5k configuration (a json).",
        default="g5k_config/small_test_run.json",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="If the job is run on a remote server. Disables data downloading.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run with debug. Add prints, launch job manually if on 1 machine.",
    )

    args = parser.parse_args()

    G5K_CONFIG_PATH = args.g5k_config
    IS_REMOTE = args.remote
    DEBUG = args.debug

    with open(G5K_CONFIG_PATH) as f:
        g5k_config = json.load(f)

    decentralizepy_config_name = g5k_config["CONFIG_NAME"]
    decentralizepy_config_path = "run_configuration/" + decentralizepy_config_name
    with open(decentralizepy_config_path) as decentralizepy_config:
        config_content_lines = decentralizepy_config.readlines()
        decentralizepy_config_content = "".join(config_content_lines)
        print(decentralizepy_config_content)
    provider, duration = launch_experiment(
        g5k_config=g5k_config,
        decentralizepy_config=decentralizepy_config_content,
        debug=DEBUG,
        is_remote=IS_REMOTE,
    )
