#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 22:18:29 2023

@author: qijingzhao
"""

import time
from functools import wraps

def timer(func):
    """
    A decorator that prints the execution time of the function it decorates.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Formatting the elapsed time nicely
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
        print(f"\nExecution time of {func.__name__}: {time_str} (hh:mm:ss.ss)")

        return result
    return wrapper
