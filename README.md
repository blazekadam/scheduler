# **TensorFlow training scheduler**

TT scheduler is a simple tool facilitating effective GPU utilization for TensorFlow training(s) in local multi-GPU environment.

- auto `CUDA_VISIBLE_DEVICES` masking
- schedule arbitrary number of trainings in advance
- simple command line usage

## Quick start

```bash
alias sch=/path/to/scheduler.py
sch --init [NUM_GPUS]
sch task.py --your_arg value
sch -n 2 training.py  # training on two GPUs
sch -f "[1,3]" training.py  # force to use GPUs 1 and 3
```

Run `sch --help` for help.

If no GPUs are available, the task will be executed as soon as possible.

## Requirements

</ul>
<li>Unix platform (tested on Ubuntu 16.04.1 LTS)</li>
<li><a href="https://www.python.org/">Python 3</a></li>
</ul>

## Usage

```                                                                
usage: scheduler.py [-h] [-i INIT] [-n NUM] [-p PREFER] [-f FORCE] [-s]       
                    [-r RELEASE [RELEASE ...]] [--cx]                         
                    [task [task ...]]                                         
                                                                              
positional arguments:                                                         
  task                  The task to run as soon as the required GPUs are      
                        available.                                            
                                                                              
optional arguments:                                                           
  -h, --help            show this help message and exit                       
  -i INIT, --init INIT  The number of available GPUs.                         
  -n NUM, --num NUM     The number of required GPUs.                          
  -p PREFER, --prefer PREFER                                                  
                        Instruct the scheduler to prefer the specified GPU(s).
  -f FORCE, --force FORCE                                                     
                        Force the scheduler to use the specified GPU(s).      
  -s, --status          Show GPU usage status (user/GPU/taskPID/start)        
  -r RELEASE [RELEASE ...], --release RELEASE [RELEASE ...]                   
                        Releases the specified GPU(s).                        
  --cx                  Append model.n_gpus=[NUM] to the task args.           
```

## Contributing
You are welcome to participate in development of this project.

## License
Scheduler is distributed under the MIT License. 
