#!/usr/bin/python3

import os
import sys
import time
import json
import fcntl
import errno
import signal
import argparse
import datetime
import threading
import subprocess

# CONSTANTS - Scheduler settings
SEC_DELAY = 3
GPU_INFO_FILE = os.path.join('/tmp', "gpu_scheduler_info")
KILL_DELAY_SEC = 3

# CONSTANTS - Data keys
GPU_AVAIL = 'avail'
GPU_USER = 'user'
GPU_TASK = 'task'
GPU_TASK_PID = 'task_pid'
GPU_TASK_START = 'task_start'
GPU_NAME = 'gpu_name'

# CONSTANTS
KILL = 0
TERMINATE = 1
WARN = 2

# GLOBAL VARIABLES
TASK_SIGNAL = TERMINATE


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--init", type=int, help="The number of available GPUs.")
    parser.add_argument("-n", "--num", type=int, default=1, help="The number of required GPUs.")
    parser.add_argument("-p", "--prefer", type=str, help="Instruct the scheduler to prefer the specified GPU(s).")
    parser.add_argument("-f", "--force", type=str, help="Force the scheduler to use the specified GPU(s).")
    parser.add_argument("-s", "--status", action='store_true', help="Show GPU usage status (user/GPU/taskPID/start)")
    parser.add_argument("-r", "--release", type=int, nargs='+', help="Releases the specified GPU(s).")
    parser.add_argument("--cx", action='store_true', help="Append model.n_gpus=[NUM] to the task args.")
    parser.add_argument("task", nargs='*', help="The task to run as soon as the required GPUs are available.")
    return parser.parse_args()


def sanitize_arg(arg):

    sanitized = json.loads(arg)
    if isinstance(sanitized, int):
        sanitized = [sanitized]

    return sanitized


# main function
def run_task(gpu_info_file, args):
    is_waiting = False

    while True:
        try:
            lock_file(gpu_info_file)
            free_gpus = get_free_gpus(gpu_info_file)
            if len(free_gpus) >= args.num:
                try:
                    if args.prefer is not None:
                        free_gpus = get_preferred_gpu(free_gpus, sanitize_arg(args.prefer))

                    if args.force is not None:
                        forced_gpus = sanitize_arg(args.force)
                        free_gpus = get_preferred_gpu(free_gpus, forced_gpus)
                        forced_gpu_free = check_forced_free(free_gpus, forced_gpus)
                        if not forced_gpu_free:
                            if not is_waiting:
                                is_waiting = True
                                print("Scheduler (PID: {}) is waiting for GPU(s) {}.".format(os.getpid(), args.force))
                            continue

                    # select required count of free gpu, which will be passed to the task
                    gpus = free_gpus[0:args.num]

                    # lock used gpu
                    set_occupied_gpu(gpu_info_file, gpus)

                    unlock_file(gpu_info_file)

                    # set enviromental variable CUDA_VISIBLE_DEVICES to comma separated list of GPU IDs
                    visible_devices = set_env_vars(gpus)

                    dt_before = datetime.datetime.now()

                    task = args.task

                    # run required task
                    if args.cx:
                        task.append('model.n_gpus={}'.format(len(gpus)))
                    p = subprocess.Popen(task, preexec_fn=before_new_subprocess)

                    # The second Ctrl-C kill the subprocess
                    signal.signal(signal.SIGINT, lambda signum, frame: stop_subprocess(p, gpu_info_file, gpus))

                    set_additional_info(gpu_info_file, gpus, os.getlogin(), task, p.pid,
                                        get_formated_dt(dt_before), visible_devices)

                    print("GPU: {}\nSCH PID: {}\nTASK PID: {}".format(visible_devices, os.getpid(), p.pid))
                    print("SCH PGID: {}\nTASK PGID: {}".format(os.getpgid(os.getpid()), os.getpgid(p.pid)))
                    p.wait()

                    dt_after = datetime.datetime.now()

                    print("\ntask: {}\nstart: {}\nend: {}\ntotal time: {}\n".format(
                          task, get_formated_dt(dt_before), get_formated_dt(dt_after),
                          get_time_duration(dt_before, dt_after)))

                    break

                # make sure the GPU is released even on interrupts
                finally:
                    set_free_gpu(gpu_info_file, free_gpus)
                    unlock_file(gpu_info_file)
                    time.sleep(1)
            else:
                unlock_file(gpu_info_file)
                time.sleep(SEC_DELAY)

        except IOError as e:
            handle_io_error(e)


def before_new_subprocess():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.setsid()


def stop_subprocess(process, gpu_file, gpu_to_release):
    """
    This function take care of the Ctrl-C (SIGINT) signal.
    On the first Ctrl-C the warning is printed.
    On the second Ctrl-C the task is terminated.
    On the third Ctrl-C the task is killed.
    Delay between terminate and kill is specified in KILL_DELAY_SEC.
    """
    def allow_kill_task():
        global TASK_SIGNAL
        TASK_SIGNAL = KILL

    def check_process_liveness(process, max_time):
        if max_time <= 0 or (process.poll() is not None):
            allow_kill_task()
        else:
            threading.Timer(0.1, lambda: check_process_liveness(process, max_time - 0.1)).start()

    global TASK_SIGNAL

    if TASK_SIGNAL is KILL:
        pgid = os.getpgid(process.pid)
        print("\nThe task (PGID: {}) was killed.".format(pgid))
        set_free_gpu(gpu_file, gpu_to_release)
        os.killpg(pgid, signal.SIGKILL)
        TASK_SIGNAL = None

    elif TASK_SIGNAL is TERMINATE:
        pgid = os.getpgid(process.pid)
        print("\nThe task (PGID: {}) was terminated.".format(pgid))
        set_free_gpu(gpu_file, gpu_to_release)

        os.killpg(pgid, signal.SIGTERM)
        # send a second SIGTERM because of blocks
        os.killpg(pgid, signal.SIGTERM)

        check_process_liveness(process, KILL_DELAY_SEC)
        TASK_SIGNAL = None

    elif TASK_SIGNAL is WARN:
        pgid = os.getpgid(process.pid)
        print("\nNext Ctrl-C terminate the task (PGID: {}).".format(pgid))
        TASK_SIGNAL = TERMINATE


def check_forced_free(gpu_indices, forced):
    if gpu_indices:
        return set(gpu_indices[:len(forced)]) == set(forced)
    return False


def get_preferred_gpu(gpu_indices, preferred):
    """Move preferred GPU on a first position if it is available."""
    for gpu in reversed(preferred):
        if gpu in gpu_indices:
            gpu_indices.remove(gpu)
            gpu_indices = [gpu, ] + gpu_indices
    return gpu_indices


# decorators
def access_gpu_file(func):
    def wrapper(f, *args, **kwargs):
        while True:
            try:
                lock_file(f)
                func(f, *args, **kwargs)
                unlock_file(f)
                break
            except IOError as e:
                handle_io_error(e)
    return wrapper


def seek_to_start(func):
    def wrapper(f, *args, **kwargs):
        f.seek(0)
        result = func(f, *args, **kwargs)
        f.seek(0)
        return result
    return wrapper


@access_gpu_file
@seek_to_start
def init_gpu_info_file(file, gpu_count):
    file.truncate()

    data = {}
    data[GPU_AVAIL] = [True]*gpu_count
    data[GPU_USER] = [None]*gpu_count
    data[GPU_TASK] = [None]*gpu_count
    data[GPU_TASK_PID] = [None]*gpu_count
    data[GPU_TASK_START] = [None]*gpu_count
    data[GPU_NAME] = [None]*gpu_count

    json.dump(data, file, indent=4, sort_keys=True)


@seek_to_start
def get_free_gpus(gpu_info_file):
    """"Returns list of GPU indices which are available."""
    gpu_states = json.load(gpu_info_file)[GPU_AVAIL]
    return [i for i, avail in enumerate(gpu_states) if avail]


@seek_to_start
def update_gpu_info(f, release_gpu, indices,
                    user=None, task=None,
                    proc_pid=None, start=None, gpu_name=None):
    gpu_data = json.load(f)
    f.seek(0)
    f.truncate()

    for i in range(len(gpu_data[GPU_AVAIL])):
        if i in indices:
            gpu_data[GPU_AVAIL][i] = release_gpu
            gpu_data[GPU_USER][i] = user
            gpu_data[GPU_TASK][i] = task
            gpu_data[GPU_TASK_PID][i] = proc_pid
            gpu_data[GPU_TASK_START][i] = start
            gpu_data[GPU_NAME][i] = gpu_name

    json.dump(gpu_data, f, indent=4, sort_keys=True)


@access_gpu_file
def set_additional_info(f, gpu_indices, user, task, proc_pid, start, gpu_name):
    update_gpu_info(f, False, gpu_indices, user, task, proc_pid, start, gpu_name)


def set_occupied_gpu(f, occupied_gpu):
    """Locks currently unavailable GPUs."""
    update_gpu_info(f, False, occupied_gpu)


@access_gpu_file
def set_free_gpu(f, free_gpu):
    """Releases GPUs"""
    update_gpu_info(f, True, free_gpu)


def get_formated_dt(dt):
    """Returns the datetime object formated."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def get_time_duration(before, after):
    """Returns the difference between two datetime objects in format: hours:minutes:seconds"""
    total_seconds = (after - before).seconds
    mins, secs = divmod(total_seconds, 60)
    hours, mins = divmod(mins, 60)
    return "{}:{}:{}".format(hours, mins, secs)


def lock_file(f):
    """Locks the file."""
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)


def unlock_file(f):
    """Unlocks the file."""
    fcntl.flock(f, fcntl.LOCK_UN)


def handle_io_error(e):
    if e.errno != errno.EAGAIN:
        raise e
    time.sleep(0.1)


def set_env_vars(gpu_indices):
    """Sets enviromental variable GPU"""
    # currently is cupported just one gpu on task
    visible_devices = ",".join([str(i) for i in gpu_indices])
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
    return visible_devices


@seek_to_start
def display_status(f):
    gpu_data = json.load(f)
    occupied = [i for i, avail in enumerate(gpu_data[GPU_AVAIL]) if not avail]
    free = [i for i, avail in enumerate(gpu_data[GPU_AVAIL]) if avail]
    if occupied:
        print("Currently used GPU:")
        print("-------------------")
        for i in occupied:
            print("GPU: {}\nUser: {}\nTask: {}\nTask PID: {}\nStarted: {}\n".format(gpu_data[GPU_NAME][i],
                                                                                    gpu_data[GPU_USER][i],
                                                                                    gpu_data[GPU_TASK][i],
                                                                                    gpu_data[GPU_TASK_PID][i],
                                                                                    gpu_data[GPU_TASK_START][i]))
    if free:
        print("Free GPU:")
        print("---------")
        for i in free:
            print("GPU {}".format(i))
    else:
        print("No GPU available.")


# run scheduler
if __name__ == '__main__':

    args = get_args()
    init = not(os.path.isfile(GPU_INFO_FILE))
    if init and not args.init:
        print('The scheduler needs to be initializes with --init arg first.')
        sys.exit(1)

    with open(GPU_INFO_FILE, 'w+' if init else 'r+') as f:
        if init:
            os.fchmod(f.fileno(), 0o777)
            init_gpu_info_file(f, args.init)

        if args.release:
            set_free_gpu(f, args.release)

        if args.status:
            display_status(f)

        if args.task:
            run_task(f, args)
