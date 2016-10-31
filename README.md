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

## Usage

1. Configure `gpu_info` file with `--init` option if needed (by default four free GPUs are created when you first run the scheduler).
2. Go to the folder where your script is stored: `cd /file/with/task`.
3. Run scheduler: e.g `../scheduler.py -t "test1.py -foo 1 -v --bar 0" --gpu_count 2`.

## Notes

* Scheduler uses just the `gpu_info` file (must be configured, see above) to store information about GPUs availability, so does not need access to GPUs driver.
* Data from stdout and stderr are stored in the directory where scheduler was run (in directory named according current date).
* Scheduler appends `--gpu` argument with allocated GPUs ID to the end of the task. e.g `test1.py -foo 1 -v --bar 0 --gpu 0 1`.

## Examples

See `./tests/run_tests` script.
