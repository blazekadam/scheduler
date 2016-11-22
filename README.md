# **Scheduler**

Scheduler is a tool for effective training Machine Learning models on multiple GPUs. 
<p>
Scheduler uses the file `gpu_scheduler_info` which is stored in /tmp/ and stores the global state about GPUs. This file is automatically created on a first start and by default is initialized for four GPU (it can be change with --init option). When a particular task is assigned to some GPU, a global variable GPU is set to `cuda[number of assigned GPU]` and this GPU is blocked until the task finishes.
</p>

## Requirements

</ul>
<li>Unix platform (not tested on Windows)</li>
<li><a href="https://www.python.org/">Python 3</a></li>
</ul>

## Example 

```
/path/to/scheduler.py "train.py foo --bar 5" --forced_gpu 2 --verbose
```
In a above example the task `train.py foo --bar 5` will be started as soon as GPU 2 will be free. 

## Contributing
You are welcome to participate in development of this project.

## License
Scheduler is distributed under the MIT License. 
