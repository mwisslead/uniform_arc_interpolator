import numpy as np

from scipy.interpolate import UnivariateSpline, PchipInterpolator, Akima1DInterpolator
import scipy.integrate as integrate
import scipy.optimize as optimize
    
def arclen_op(t, splines):
    return np.sqrt(np.sum([np.square(spline(t)) for spline in splines]))

def uniform_arc_interpolator(pts, method='spline', degree=3):
    '''return callable that given arg t in range [0, 1] gives the point
       that is the fraction t along the curve defined by pts'''

    if len(pts) < 2:
        raise ValueError('Cannot interpolate without atleast 2 points')

    methodl = method.lower()
    if methodl == 'linear':
        interpfunc = lambda x, y: UnivariateSpline(x, y, k=1, s=0)
    elif methodl == 'spline':
        interpfunc = lambda x, y: UnivariateSpline(x, y, s=0, k=degree)
    elif methodl == 'pchip':
        interpfunc = PchipInterpolator
    elif methodl == 'akima':
        interpfunc = Akima1DInterpolator
    else:
        raise ValueError('invalid method {}'.format(repr(method)))

    pts = np.array(pts)
   
    t = np.arange(len(pts))
    splines = [interpfunc(t, pts[:, i]) for i in range(pts.shape[1])]

    splinesd = [spline.derivative() for spline in splines]

    def arclen_piece(tp):
        return arclen_op(tp, splinesd)

    def arclen_func(tp):
        return integrate.quad(arclen_piece, 0, tp)[0]

    arclens = np.array([arclen_func(tp) for tp in t])
    arclen = arclens[-1]

    def arclen_func(tp):
        x0 = min(int(tp), len(arclens)-1)
        y0 = arclens[x0]
        return y0 + integrate.quad(arclen_piece, x0, tp)[0]

    def min_func(x, tp):
        return np.abs(tp*arclen - arclen_func(x))

    def c(tp):
        x = np.array([optimize.fmin(min_func, p*arclen, args=(p,)) for p in tp])
        return _interpolator(x, splines)

    return c

def _interpolator(t, spl):
    return np.concatenate(tuple([spli(t)] for spli in spl))

def main():
    import matplotlib.pyplot as plt
    x = np.linspace(0, 4*np.pi, 17)
    y = np.sin(x)
    plt.plot(x, y, '.')
    plt.show()
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    pts = np.concatenate(([(x-xmin)/(xmax-xmin)], [(y-ymin)/(ymax-ymin)]), axis=0).T
    f = uniform_arc_interpolator(pts, 'spline')
    t = np.linspace(0, 1, 65)
    x, y = f(t)
    x = (xmax-xmin)*x + xmin
    y = (ymax-ymin)*y + ymin
    plt.plot(x, y, '.')
    plt.show()

if __name__ == '__main__':
    main()
