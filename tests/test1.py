#!/usr/bin/env python3

import os
import signal
import argparse
import ast
import sys

print(sys.argv)

p = argparse.ArgumentParser()
p.add_argument("--port_shift", type=int)
p.add_argument("--net_type cleavages")
p.add_argument("--max_hours")
p.add_argument("--min_hours")
p.add_argument("--min_exp_hours")
p.add_argument("--max_cc")
p.add_argument("--normalize", action='store_true')
p.add_argument("--equalize", action='store_true')
p.add_argument("--rotation")
p.add_argument("--flip", action='store_true')
p.add_argument("--time_shift")
p.add_argument("--input_size")
p.add_argument("--lc_architecture")
p.add_argument("--cnn_architecture")
p.add_argument("--mlp_hidden_dim")
p.add_argument("--rnn_dim")
p.add_argument("--activation")
p.add_argument("--lc_nums")
p.add_argument("--learning_rate")
p.add_argument("--batch_size")
p.add_argument("--sliding_averages", action='store_true')
p.add_argument("--long_average")
p.add_argument("--short_average")
p.add_argument("--save_freq")

a = p.parse_args()
print()
print(a)

lc = ast.literal_eval(a.lc_architecture)
cnn = ast.literal_eval(a.cnn_architecture)
print(lc['conv_sizes'], type(lc), type(cnn))

signal.signal(signal.SIGTERM, signal.SIG_IGN)
while True:
    pass

print("-------------> test1", os.getenv("GPU"))
