from sage.all import *
from util import get_datapts
import numpy as np
from random import seed
import matplotlib.pyplot as plt

XMIN, XMAX = 0, 4
YMIN, YMAX = -1, 5
ERRMIN, ERRMAX = [-1], [1]
POINT_COUNT = 5000
REGRESSAND_DEGREE = 3
assert POINT_COUNT >= REGRESSAND_DEGREE

def regression(points, dim:int=2):
    # Ax = b -> canot be solved for x. But:
    # A^tAx = A^tb represents a projection into solution space, which has a solution, so:
    # x = (A^tA)^-1 * A^tb

    # for linear:
    # f(x) = c1 + c2x
    # Ax = b ->
    # (1 x1) * (c1) = (y1) 
    # (1 x2) * (c2)   (y2)
    # print(points); exit()
    xs = points[0]
    ys = points[1]
    AT = np.array([xs**i for i in range(dim)])
    A = AT.transpose()
    ATA = AT @ A
    coeffs = np.linalg.inv(ATA) @ AT @ ys
    return np.polynomial.Polynomial(coeffs)

def trig_regression_ez(points):
    # f(x) = a + b * sin(x)
    xs = points[:,0]
    ys = points[:,1]
   
    AT = np.array([np.ones_like(xs), sin(xs)])
    A = AT.transpose()
    ATA = AT @ A
    a,b = np.linalg.inv(ATA) @ AT @ ys
    return lambda x: a+b*sin(x)

def trig_regression_med(points):
    # f(x) = a * sin(wx) + b * cos(ox)
    w, o = 1.2, -2.3
    xs = points[:,0]
    ys = points[:,1]
    AT = np.array([sin(w * xs), cos(o * xs)])
    A = AT.transpose()
    ATA = AT @ A
    a,b = np.linalg.inv(ATA) @ AT @ ys
    return lambda x: a+b*sin(x)

def plot_pts(pts):
    return point(pts, xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX)

plt.rcParams.update({
    "figure.facecolor":  (1.0, 0.0, 0.0, 0.0),  # red   with alpha = 30%
    "axes.facecolor":    (0.0, 1.0, 0.0, 0.0),  # green with alpha = 50%
    "savefig.facecolor": (0.0, 0.0, 1.0, 0.0),  # blue  with alpha = 20%
    "axes.edgecolor": (0,1,1,1),
    "xtick.color": (0,1,1,1),
    "ytick.color": (0,1,1,1),
    "lines.color": (1,0,1),
    "xtick.labelsize": 'large',
    "ytick.labelsize": 'large',
})

stem_func = lambda x: 0.5 * x**2
pts = get_datapts(stem_func, POINT_COUNT, 1, XMIN, XMAX, ERRMIN, ERRMAX, seed=0)
x_pts, y_pts = pts
regression_pts = regression(pts, REGRESSAND_DEGREE)(x_pts)
plt.xlim((XMIN,XMAX))
plt.ylim((YMIN, YMAX))
axes:plt.Axes = plt.gca()
axes.set_facecolor('None')
axes.set_alpha(0)
plt.plot(*pts, 'o', color=(1,0,1))
#plt.plot(x_pts, stem_func(x_pts))
plt.plot(x_pts, regression(pts, REGRESSAND_DEGREE)(x_pts), color=(0,1,0))
#for i in range(len(pts[0])):
xx = np.vstack([x_pts,x_pts])
yy = np.vstack([y_pts,regression_pts])
#plt.plot(xx ,yy, ".-.", color=(1,.5,0))
plt.savefig('/tmp/regression_pts')
# plt.plot(x_pts, trig_regression_ez(pts)(x_pts))
# plt.plot(x_pts, trig_regression_med(pts)(x_pts))
plt.show()
