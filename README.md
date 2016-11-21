# Scheduler

## Options

Show help with: `./scheduler.py -h`

~~~~
    usage: scheduler.py [-h] [-t TASK] [-gc GPU_COUNT] [-i INIT [INIT ...]]
    
    optional arguments:
      -h, --help            show this help message and exit
      -t TASK, --task TASK  The quoted task with arguments which will be started on free
                            GPUs as soon as possible.
      -gc GPU_COUNT, --gpu_count GPU_COUNT
                            The count of required GPUs for specified task.
      -i INIT [INIT ...], --init INIT [INIT ...]
                            Initializes gpu_info file. List of numbers is
                            expected, where the first number is total count of GPUs
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

 
~~~~
