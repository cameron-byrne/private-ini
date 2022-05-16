import multiprocessing as mp
import time
import psutil
import numba

def main():
    # change this to change the number of cores being used
    num_cores = 60

    timer = Timer()

    timer.start()
    print("vanilla: ")
    add_nums()
    timer.stop()


    timer.start()

    # complete a task of adding a ton of numbers together
    for i in range(num_cores):
        add_nums()
    print("single-core processing")
    timer.stop()

    '''
    timer.start()
    q = mp.Queue()
    processes = []
    
    # complete the same task but split across several processes
    for core in range(num_cores):
        p = mp.Process(target=add_nums_multi, args=(q,))
        p.start()
        processes.append(p)

    # this blocks the program until each process completes
    for k in range(num_cores):
        processes[k].join()

    print("")
    print(f"{num_cores} core processing")
    timer.stop()
    '''

    timer.start()
    with mp.Pool(num_cores) as p:
        p.map(add_nums, range(num_cores))
    print("")
    print(f"{num_cores} core processing using pooling instead")
    timer.stop()

    timer.start()
    with mp.Pool(num_cores) as p:
        p.map(add_nums_fast, range(num_cores))
    print("")
    print(f"{num_cores} core processing using numba and pooling")
    timer.stop()

    timer.start()
    a = add_nums_parallel()
    print("")
    print(f"{num_cores} core processing using numba's parallelization")
    timer.stop()


def add_nums(dummy=0):
    sum = 0
    for k in range(10000000):  # ten million
        sum += k
    return sum


@numba.njit()
def add_nums_fast(dummy=0):
    sum = 0
    for k in range(10000000):  # ten million
        sum += k
    return sum

@numba.njit(parallel=True)
def add_nums_parallel(dummy=0):
    sum = 0
    for i in numba.prange(7):
        for k in range(10000000):  # ten million
          sum += k
    return sum

def add_nums_multi(q):
    sum = 0
    for k in range(20000000):  # ten million
        sum += k
    q.put(sum)


# not my code, this is just to time things
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