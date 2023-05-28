import numpy as np
from random import uniform, seed as set_seed
import matplotlib.pyplot as plt

XMIN, XMAX = 0, 4

def get_datapts(func, c:int, n:int, xmin, xmax, errmin, errmax, seed:int|None=None):
    '''Returns an array with c columns and n rows, i.e. c points of dimension n.'''
    if isinstance(seed, int):
        set_seed(seed)
    datamatrix = np.ndarray((n+1,c))
    datamatrix[0] = np.linspace(xmin, xmax, c)
    datamatrix[1] = func(datamatrix[0])
    ynoise = np.fromiter((uniform(errmin[i], errmax[i]) for i in range(n) for _ in range (c)), float, count=c*n).reshape((n,c))
    datamatrix[1:,] += ynoise
    return datamatrix

def plot_func(f,**kwargs):
    xs = np.linspace(XMIN, XMAX, 1000)
    ys = f(xs)
    print(xs, ys)
    plt.plot(xs, ys, **kwargs)
#wtf