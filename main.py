from regression import regression
from util import Plotter
from numpy import vstack

POINT_COUNT = 10
REGRESSAND_DEGREE = 3
assert POINT_COUNT >= REGRESSAND_DEGREE

DRAW_PTS = True
DRAW_STEM = True
DRAW_REGRESSION = True
DRAW_ERRORS = True

def main():
  plotter = Plotter()
  stem_func = lambda x: 0.5 * x**2
  pts = plotter.get_datapts(stem_func, POINT_COUNT, 1, seed=0)
  if DRAW_PTS: plotter.plot_pts(*pts, 'o', color=(1,0,1))
  if DRAW_STEM: plotter.plot_func(stem_func)
  if DRAW_REGRESSION: plotter.plot_func(regression(pts, REGRESSAND_DEGREE), color=(0,1,0))
  if DRAW_ERRORS:
    xx = vstack([pts[0], pts[0]])
    yy = vstack([pts[1], regression(pts, REGRESSAND_DEGREE)(pts[0])])
    plotter.plot_pts(xx ,yy, ".-.", color=(1,.5,0))
    reg_func = regression(pts ,REGRESSAND_DEGREE)
    plotter.measure_error(reg_func, pts[1])
    
  plotter.save('/tmp/regression_pts')
  plotter.show()

if __name__ == "__main__":
  main()