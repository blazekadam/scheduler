#!/usr/bin/env python3

import argparse
import json
import time
import subprocess
import fcntl
import errno
import datetime
import os
import sys
import signal
import threading
import shlex

# CONSTANTS - Scheduler settings
SEC_DELAY = 3
PATH = "/tmp/"
GPU_INFO_FILE = os.path.join(PATH, "gpu_scheduler_info")
DEFAULT_GPU_COUNT = 4
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
TASK_SIGNAL = WARN


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gc", "--gpu_count", type=int, default=1,
                        help="The count of required GPUs for specified task.")
    parser.add_argument("-i", "--init", nargs="+", type=int,
                        help="""Initializes gpu info file. List of numbers is expected,
                        where first number is total count of GPUs and the rest of the numbers denotes unavailable GPUs.
                        e.g -i 5 3 4 means that total count of GPUs is 5 and GPU 3 and 4 are currently unavailable.""")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Prints info about the process, when the task is completed.")
    parser.add_argument("-o", "--out", nargs="?", type=argparse.FileType('w'), default=sys.stdout,
                        help="The name of the file, which will be used to store stdout. The default file is sys.stdout.")
    parser.add_argument("-e", "--err", nargs="?", type=argparse.FileType('w'), default=sys.stderr,
                        help="The name of the file, which will be used to store stderr. The default file is sys.stderr.")
    parser.add_argument("-pg", "--prefered_gpu", type=int,
                        help="If possible, prefered GPU is assigned to the task, otherwise is assigned random free GPU.")
    parser.add_argument("-fg", "--forced_gpu", type=int,
                        help="Wait until specified GPU is free.")
    parser.add_argument("-s", "--status", action='store_true',
                        help="Show info about GPU usage - user/GPU/taskPID/start")
    parser.add_argument("-rg", "--release_gpu", type=int, nargs='+',
                        help="Releases GPUs according their indices. e.g -rg 0 2 will release GPU 0 and 2.")
    parser.add_argument("task", nargs='?',
                        help="The quoted task with arguments which will be started on free GPUs as soon as possible.")
    return parser.parse_args()


# main function
def run_task(gpu_info_file, args):
    is_waiting = False

    while True:
        try:
            lock_file(gpu_info_file)
            free_gpu = get_free_gpu(gpu_info_file)
            if len(free_gpu) >= args.gpu_count:

                try:
                    if args.prefered_gpu is not None:
                        free_gpu = get_prefered_gpu(free_gpu, args.prefered_gpu)

                    if args.forced_gpu is not None:
                        free_gpu = get_prefered_gpu(free_gpu, args.forced_gpu)
                        forced_gpu_free = check_forced_free(free_gpu, args.forced_gpu)
                        if not forced_gpu_free:
                            if not is_waiting:
                                is_waiting = True
                                print("Scheduler (PID: {}) is waiting for GPU {}.".format(os.getpid(), args.forced_gpu))
                            continue

                    # select required count of free gpu, which will be passed to the task
                    free_gpu = free_gpu[0:args.gpu_count]

                    # lock used gpu
                    set_occupied_gpu(gpu_info_file, free_gpu)

                    unlock_file(gpu_info_file)

                    # set enviromental variable GPU to cuda[index of allocated GPU]
                    cuda = set_env_vars(free_gpu)

                    dt_before = datetime.datetime.now()

                    # parse string of args to list
                    task = prepare_args(args.task)

                    # replace char '#' with port number
                    task = insert_portshift(task, free_gpu[0])

                    # run required task
                    p = subprocess.Popen(task,
                                         stdout=args.out,
                                         stderr=args.err,
                                         preexec_fn=before_new_subprocess)

                    # The second Ctrl-C kill the subprocess
                    signal.signal(signal.SIGINT, lambda signum, frame: stop_subprocess(p, gpu_info_file, free_gpu))

                    set_additional_info(gpu_info_file, free_gpu, os.getlogin(), task,
                                        p.pid, get_formated_dt(dt_before), cuda)

                    print("GPU: {}\nSCH PID: {}\nTASK PID: {}".format(cuda, os.getpid(), p.pid))
                    print("SCH PGID: {}\nTASK PGID: {}".format(os.getpgid(os.getpid()), os.getpgid(p.pid)))
                    p.wait()

                    dt_after = datetime.datetime.now()

                    # info message
                    if args.verbose:
                        print("\ntask: {}\nstdout: {}\nstderr: {}\nstart: {}\nend: {}\ntotal time: {}\n".format(
                            task, args.out.name, args.err.name,
                            get_formated_dt(dt_before), get_formated_dt(dt_after),
                            get_time_duration(dt_before, dt_after)))

                    break

                # make sure the GPU is released even on interrupts
                finally:
                    set_free_gpu(gpu_info_file, free_gpu)
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


def prepare_args(args):
    result = []
    for a in args.split('\n'):
        if a != '':
            result.extend(shlex.split(a))
    return result


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
        return gpu_indices[0] == forced
    return False


def get_prefered_gpu(gpu_indices, prefered):
    """Move prefered GPU on a first position if it is available."""
    if prefered in gpu_indices:
        gpu_indices.remove(prefered)
        return [prefered, ] + gpu_indices
    return gpu_indices


def insert_portshift(task, task_id):
    port = 3600 + task_id * 100
    task = list(map(lambda v: str(port) if v == '__num__' else v, task))
    return task


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
def init_gpu_info_file(f, gpu_count, occupied_gpu):
    """
    occupied_gpu - indices of GPUs which currently are not available
    gpu_count - total count of GPUs on a system
    """
    gpu_states = [False if i in occupied_gpu else True for i in range(gpu_count)]
    f.truncate()
    data = {}
    data[GPU_AVAIL] = gpu_states
    init_to_none = lambda c: c * [None]
    data[GPU_USER] = init_to_none(gpu_count)
    data[GPU_TASK] = init_to_none(gpu_count)
    data[GPU_TASK_PID] = init_to_none(gpu_count)
    data[GPU_TASK_START] = init_to_none(gpu_count)
    data[GPU_NAME] = init_to_none(gpu_count)
    json.dump(data, f, indent=4, sort_keys=True)


@seek_to_start
def get_free_gpu(gpu_info_file):
    "Returns list of GPU indices which are available."
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
    cuda = "cuda{}".format(gpu_indices[0])
    os.environ['GPU'] = cuda
    return cuda


def validate_args(args):
    if args.gpu_count != 1:
        print("Usage of multiple GPUs isn't supported yet. You must use just the one GPU for the task.")
        sys.exit(1)


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

    mode = 'r+'
    need_init_gpuf = not(os.path.isfile(GPU_INFO_FILE))
    if need_init_gpuf:
        mode = 'w+'

    with open(GPU_INFO_FILE, mode) as f:
        if need_init_gpuf:
            init_gpu_info_file(f, DEFAULT_GPU_COUNT, [])

        # parse cli args
        args = get_args()
        validate_args(args)

        if args.init:
            init_gpu_info_file(f, args.init[0], args.init[1:])

        if args.release_gpu:
            set_free_gpu(f, args.release_gpu)

        if args.status:
            display_status(f)

        if args.task:
            run_task(f, args)
