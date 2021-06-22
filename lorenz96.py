import numpy as np

class Lorenz96:
    def __init__(self, k: int, f, dt):
        self.k = k
        self.f = f
        self.dt = dt
        self.idxn2 = list( range(-2, k-2) )
        self.idxn1 = list( range(-1, k-1) )
        tmp = list( range(1, k+1) )
        tmp[-1] = 0
        self.idxp1 = tmp
        tmp = list( range(2, k+2) )
        tmp[-2] = 0
        tmp[-1] = 1
        self.idxp2 = tmp

#    def init(self, const, sigma, nspin = 100):
    def init(self, const, sigma, nspin = 5000):
        x = np.random.randn(self.k) * sigma + const
        for n in range(nspin):
            x = self.forward(x)
        return x

    def __derivative(self, x):
        dx = x[self.idxn1] * ( x[self.idxp1] - x[self.idxn2] ) - x + self.f
        return dx

    def forward(self, x):
        d1 = self.__derivative(x)
        d2 = self.__derivative(x + d1 * ( self.dt * 0.5 ) )
        d3 = self.__derivative(x + d2 * ( self.dt * 0.5 ) )
        d4 = self.__derivative(x + d3 * self.dt)
        x = x + ( d1 + ( d2 + d3 ) * 2.0 + d4 ) * ( self.dt / 6.0 )
        return x

    def __derivative_ad(self, df, x):
        # y = x[self.idxn1] * ( x[self.idxp1] - x[self.idxn2] ) - x + self.f
        dydx = ( x[self.idxp2] - x[self.idxn1] ) * df[self.idxp1] \
               + x[self.idxn2] * df[self.idxn1] \
               - x[self.idxp1] * df[self.idxp2] \
               - df
        return dydx

    def adjoint(self, ddx, x):
        # y = x + ( d1 + d2 * 2.0 + d3 * 2.0 + d4 ) * ( self.dt / 6.0 )
        x1 = x + self.__derivative(x) * ( self.dt * 0.5 )
        x2 = x + self.__derivative(x1) * ( self.dt * 0.5 )
        x3 = x + self.__derivative(x2) * self.dt
        dydx = ddx
        d1 = ddx * ( self.dt / 6.0 )
        d2 = ddx * ( self.dt / 3.0 )
        d3 = ddx * ( self.dt / 3.0 )
        d4 = ddx * ( self.dt / 6.0 )
        x3 = self.__derivative_ad(d4, x3)
        dydx += x3
        d3 += x3 * self.dt
        x2 = self.__derivative_ad(d3, x2)
        dydx += x2
        d2 += x2 * ( self.dt * 0.5 )
        x1 = self.__derivative_ad(d2, x1)
        dydx += x1
        d1 += x1 * ( self.dt * 0.5 )
        dydx += self.__derivative_ad(d1, x)
        return dydx


if __name__ == '__main__':

    np.random.seed(0)

    k = 40
    f = 8.0
    dt = 0.01

    nt = 100
#    nt = 400

    model = Lorenz96(k, f, dt)
    x0 = model.init(f, 0.01)
#    x0 = model.init(f, 0.01, nspin=500)

    xa = np.zeros([nt+1,k])
    x = x0

    # spinup
    for n in range(100):
        x = model.forward(x)
    xa[0,:] = x
    for n in range(nt):
        x = model.forward(x)
        xa[n+1,:] = x

    import matplotlib.pyplot as plt

#    plt.imshow(xa, aspect=k/nt)
#    plt.colorbar()
#    plt.show()


#    np.random.seed(1)

    e0 = 1e-1
    ns = 1000
    err = np.zeros([nt+1,ns])
    for l in range(ns):
        y = x0 + np.random.randn(k) * e0

        # spinup
        for n in range(100):
            y = model.forward(y)
        err[0,l] = ( ( y - xa[0,:] )**2 ).mean()
        for n in range(nt):
            y = model.forward(y)
            err[n+1,l] = ( ( y - xa[n+1,:] )**2 ).mean()


    err = err.mean(axis=1)

    print(err.min(), err.max())

    plt.plot(err)
    ax = plt.gca()
    ax.set_yscale("log")

    plt.grid(which="both")

    plt.show()

