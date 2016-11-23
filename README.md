# **Scheduler**

Scheduler is a tool for effective GPU utilization. The typical usage is e.g. training many machine learning models on multiple GPUs. Using the scheduler allows you to easily schedule more tasks than your total number of GPUs.

Scheduler uses the file `gpu_scheduler_info` which is stored in `/tmp/` and contains the global GPUs state. This file is automatically created on the first start and, by default, is initialized for four GPU (it can be change with `--init` option). When a particular task is assigned to some GPU, a global variable GPU is set to `cuda[id of assigned GPU]` and this GPU is blocked until the task finishes.

## Requirements

</ul>
<li>Unix platform (tested on Ubuntu 16.04.1 LTS)</li>
<li><a href="https://www.python.org/">Python 3</a></li>
</ul>

## Example 

```
/path/to/scheduler.py "train.py foo --bar 5" --forced_gpu 2 --verbose
```
In a above example the task `train.py foo --bar 5` will be started as soon as GPU 2 is free. 

## Usage

```
usage: schedule [-h] [-gc GPU_COUNT] [-i INIT [INIT ...]] [-v] [-o [OUT]]
                [-e [ERR]] [-pg PREFERED_GPU] [-fg FORCED_GPU] [-s]
                [-rg RELEASE_GPU [RELEASE_GPU ...]]
                [task]

positional arguments:
  task                  The quoted task with arguments which will be started
                        on free GPUs as soon as possible.

optional arguments:
  -h, --help            show this help message and exit
  -gc GPU_COUNT, --gpu_count GPU_COUNT
                        The count of required GPUs for specified task.
  -i INIT [INIT ...], --init INIT [INIT ...]
                        Initializes gpu info file. List of numbers is
                        expected, where first number is total count of GPUs
                        and the rest of the numbers denotes unavailable GPUs.
                        e.g -i 5 3 4 means that total count of GPUs is 5 and
                        GPU 3 and 4 are currently unavailable.
  -v, --verbose         Prints info about the process, when the task is
                        completed.
  -o [OUT], --out [OUT]
                        The name of the file, which will be used to store
                        stdout. The default file is sys.stdout.
  -e [ERR], --err [ERR]
                        The name of the file, which will be used to store
                        stderr. The default file is sys.stderr.
  -pg PREFERED_GPU, --prefered_gpu PREFERED_GPU
                        If possible, prefered GPU is assigned to the task,
                        otherwise is assigned random free GPU.
  -fg FORCED_GPU, --forced_gpu FORCED_GPU
                        Wait until specified GPU is free.
  -s, --status          Show info about GPU usage - user/GPU/taskPID/start
  -rg RELEASE_GPU [RELEASE_GPU ...], --release_gpu RELEASE_GPU [RELEASE_GPU ...]
                        Releases GPUs according their indices. e.g -rg 0 2
                        will release GPU 0 and 2.
```

## Contributing
You are welcome to participate in development of this project.

## License
Scheduler is distributed under the MIT License. 
