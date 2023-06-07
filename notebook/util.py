import numpy as np
from random import uniform, seed as set_seed
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor":  (1.0, 0.0, 0.0, 0.0),
    "axes.facecolor":    (0.0, 1.0, 0.0, 0.0),
    "savefig.facecolor": (0.0, 0.0, 1.0, 0.0),
    "axes.edgecolor": (0,1,1,1),
    "xtick.color": (0,1,1,1),
    "ytick.color": (0,1,1,1),
    "lines.color": (1,0,1),
    "xtick.labelsize": 'large',
    "ytick.labelsize": 'large',
})


class Plotter:
    def __init__(self, xmin=0, xmax=4, ymin=0, ymax=6, errmin=None, errmax=None):
        self.ERRMIN = errmin if errmin is not None else [-1]
        self.ERRMAX = errmax if errmax is not None else [1]
        self.XMIN, self.XMAX = xmin, xmax
        self.YMIN, self.YMAX = ymin, ymax
        # self.axes = plt.axes()
        self.plot_data = []

    def plot_pts(self, xs, ys, *args, **kwargs):
        self.plot_data.append((xs, ys, args, kwargs))

    def plot_func(self, f, *args, **kwargs):
        xs = np.linspace(self.XMIN, self.XMAX, 1000)
        self.plot_data.append((xs, f(xs), args, kwargs))
        
    def get_datapts(self, func, c:int, n:int, seed=None):
        '''Returns an array with c columns and n rows, i.e. c points of dimension n.'''
        if isinstance(seed, int):
            set_seed(seed)
        datamatrix = np.ndarray((n,c))
        datamatrix[0] = np.linspace(self.XMIN, self.XMAX, c)
        datamatrix[1] = func(datamatrix[0])
        ynoise = np.fromiter((uniform(self.ERRMIN[i], self.ERRMAX[i]) for i in range(n-1) for _ in range (c)), float, count=c*(n-1)).reshape((n-1,c))
        datamatrix[1:,] += ynoise
        return datamatrix

    def measure_error(self, func, ys):
        xs = np.linspace(self.XMIN, self.XMAX, len(ys))
        func_ys = func(xs)
        error_data = (0.1, 4.9 , f"Regression Error: " + str(sum((ys - func_ys)**2)))
        self.plot_data.append(error_data)

    def show(self):
        for xs, ys, args, kwargs in self.plot_data:
            plt.plot(xs, ys, *args, **kwargs)
        plt.xlim(self.XMIN, self.XMAX)
        plt.ylim(self.YMIN, self.YMAX)
        plt.show()

    def save(self, path):
        for xs, ys, args, kwargs in self.plot_data:
            plt.plot(xs, ys, *args, **kwargs)
        plt.xlim(self.XMIN, self.XMAX)
        plt.ylim(self.YMIN, self.YMAX)
        plt.savefig(path)

    def reset(self):
        self.plot_data = []
