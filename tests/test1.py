#!/usr/bin/env python3

import os
import signal

signal.signal(signal.SIGTERM, signal.SIG_IGN)
while True:
    pass

print("-------------> test1", os.getenv("GPU"))
