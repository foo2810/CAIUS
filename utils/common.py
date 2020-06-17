import time
import sys
import contextlib

__all__ = ['time_counter']

@contextlib.contextmanager
def time_counter(r=1.):
    try:
        # s = time.time()
        s = time.perf_counter()
        yield
    finally:
        # e = time.time()
        e = time.perf_counter()
        t = (e - s) / r
        sys.stderr.write(' >>> Info: [Time (time_counter)] {:.7f}s\n'.format(t))

