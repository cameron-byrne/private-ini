import multiprocessing as mp
import time
import psutil

def main():
    timer = Timer()
    timer.start()
    for i in range(60):
        add_nums()
    timer.stop()

    timer.start()
    q = mp.Queue()
    processes = []
    for core in range(60):

        p = mp.Process(target=add_nums_multi, args=(q,))
        p.start()
        processes.append(p)
    for k in range(60):
        processes[k].join()

    print(q.qsize())
    timer.stop()


def add_nums():
    sum = 0
    for k in range(10000000):  # ten million
        sum += k
    return sum

def add_nums_multi(q):
    sum = 0
    for k in range(10000000):  # ten million
        sum += k
    q.put(sum)


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")

if __name__ == "__main__":
    main()