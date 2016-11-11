#!/usr/bin/env python3

import os
import signal
import argparse
import ast
import sys

print(sys.argv)

p = argparse.ArgumentParser()
p.add_argument("--lc_architecture")
p.add_argument("--port_shift")
a = p.parse_args()

val = a.lc_architecture
print(val)
parsed = ast.literal_eval(val)
print(parsed['conv_sizes'])

signal.signal(signal.SIGTERM, signal.SIG_IGN)
while True:
    pass

print("-------------> test1", os.getenv("GPU"))
