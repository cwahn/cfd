from abc import *
from copy import deepcopy
from functools import partial
from math import isnan
from time import sleep
from typing import Callable, List, Tuple

from more_itertools import last
from traitlets import default


class T(metaclass=ABCMeta):
    @abstractmethod
    def succ(self) -> "T":
        pass

    @abstractmethod
    def partial_diff(self, y: float, succ_y: float) -> float:
        pass

    @abstractmethod
    def update(self, pds: List[float]) -> "T":
        pass


Theta = List[T]

last = partial(last, default=[])

def succ_i(idx: int, ts: List[T]) -> List[T]:
    new_ts = deepcopy(ts)
    new_ts[idx] = new_ts[idx].succ()
    return new_ts

def gradient(f: Callable[[List[T]], float], ts: List[T]) -> Tuple[float, List[float]]:
    # print("ts of grad: ", ts)
    y = f(ts)
    ys = [y] * len(ts)

    idxs = range(len(ts))
    succ_tss = map(partial(succ_i, ts=ts), idxs)
    succ_ys = map(f, succ_tss)

    triples = zip(ts, ys, succ_ys)
    partial_diffs = map(lambda t: t[0].partial_diff(t[1], t[2]), triples)

    return (y, list(partial_diffs))

def optimize(
        f: Callable[[List[T]], float], 
        thetas : List[Theta], 
        ys: List[float],
        pdss : List[List[float]],
        stop_f: Callable[[List[Theta], List[float], List[List[float]]], bool],
        log_f: Callable[[int, Theta, float, List[float]], None]
        ) -> Tuple[List[T], float]:
    
    ts = list(map(last, thetas))

    y, new_pds = gradient(f, ts)
    # print("new_pds: ", new_pds)
    new_ys = ys + [y]
    # print("pdss: ", pdss)
    new_pdss = list(map(lambda tp: tp[0] + [tp[1]], zip(pdss, new_pds)))

    # print("ts: ", ts)
    # print("new_pdss: ", new_pdss)
    new_ts = list(map(lambda tp: tp[0].update(tp[1]) ,zip(ts, new_pdss)))

    # print("thetas: ", thetas)
    # print("new_ts: ", new_ts)
    new_thetas = list(map(lambda tp: tp[0] + [tp[1]], zip(thetas, new_ts)))
    # print("new_thetas: ", new_thetas)

    log_f(len(ys), new_ts, y, new_pds)
    sleep(0.3)

    if not stop_f(new_thetas, new_ys, new_pdss):
        # print(new_thetas)
        return optimize(f, new_thetas, new_ys, new_pdss, stop_f, log_f)
    else:
        return (list(map(last, new_thetas)), last(new_ys))
    