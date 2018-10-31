import numpy as np
import matplotlib.pyplot as plt
import time

class Line():
    def __init__(self):
        # we need 2 coordinates, length and angle to represent a rod
        self.length = 1
        self.m = 1
        self.d_t = .01  # s
        
        self.theta = 0
        self.omega = 0  # angular velocity
        
        start = np.array([0, 0])
        
        self.center = start + .5*self.length*self.dir()
        self.v = np.array([0, 0])
        
    def get_end(self):
        return self.center + .5*self.length * self.dir()
        # return [self.start[0] + self.length * np.cos(self.theta), self.start[1] + self.length * np.sin(self.theta), ]
    
    def get_start(self):
        return self.center - .5*self.length * self.dir()

    def dir(self, theta=None):
        if theta is None:
            return np.array([np.cos(self.theta), np.sin(self.theta)])
        else:
            return np.array([np.cos(theta), np.sin(theta)])

    def plot(self):
        start = self.get_start()
        end = self.get_end()

        plt.plot([start[0], end[0]], [start[1], end[1]],'-o')

    def gravity(self):
        g = 9.81    # m/s**2

        F_g = np.array([0, -g])
        
        F_g_par = self.dir()*np.dot(self.dir(), F_g)
        F_g_perp = F_g - F_g_par

        F_0_init = -F_g_par - .25 * F_g_perp
    
        lst_F_init = [F_g,
                      F_0_init
                      ]
        lst_x_init =  [self.center,#.5*self.length*self.dir(),
                       np.array([0, 0])
                       ]

        theta_next, _ = self.predict_theta(lst_F_init, lst_x_init)
        dir1 = self.dir(theta=theta_next)

        F_g1_par = dir1*np.dot(dir1, F_g)
        F_g1_perp = F_g - F_g1_par
        
        F_0_next = (-F_g1_par -.25*F_g1_perp)

        # TODO average/integrate over theta

        # F_0 = self.dir() * np.dot(self.dir(), F_g)
        # F_0 = np.array([0, 0])  # Origin not fixed
        # F_0 = -F_g_par - .25 * F_g_perp
        # TODO, this one seems the most promising (predict the next, then take average)
        F_0 = .5*(F_0_init + F_0_next)     # average

        lst_F = [F_g,
                 F_0
                 ]
        lst_x =  [self.center,#.5*self.length*self.dir(),   # TODO center averaged with predicted one
                 np.array([0, 0])
                 ]

        self.update(lst_F, lst_x)

    def predict_theta(self, lst_F, lst_x):
        # rotation
        theta0 = self.theta + 0
        omega0 = self.omega + 0

        def cross2d(v1, v2):
            return v1[0] * v2[1] - v1[1] * v2[0];

        tau = 0
        for F_i, x_i in zip(lst_F, lst_x):
            r_i = x_i - self.center
            tau += cross2d(r_i, F_i)

        I = 1/12*self.m*self.length**2
        alpha = tau / I
        
        d_theta = omega0 * self.d_t + .5 * alpha * (self.d_t ** 2)
        d_omega = alpha * self.d_t

        theta1 = theta0 + d_theta
        omega1 = omega0 + d_omega
        return theta1, omega1
        
    def update(self, lst_F, lst_x):
        # linear motion
        a_tot = sum(lst_F)/self.m
        
        x0 = self.center.copy()
        v0 = self.v.copy()

        d_x = v0*self.d_t + .5*a_tot*(self.d_t**2)
        d_v = a_tot*self.d_t

        self.center = x0 + d_x
        self.v = v0 + d_v
        
        # rotation
        self.theta, self.omega = self.predict_theta(lst_F, lst_x)


class Lines():
    def __init__(self):
        self.line = Line()
    # TODO
    
    def next(self):
        ...
        # TODO

        self.line.gravity()
    
    def sim(self, n=1):
        for i in range(n):
            self.next()
            self.plot()

            time.sleep(.1)

    def plot(self):
        plt.figure(1)
        # plt.figure()
    
        plt.axes(aspect='equal')
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        
        self.line.plot()
        
        plt.show(0)
        

def main():
    lines = Lines()
    
    lines.sim(20)
    
    
if __name__ == '__main__':
    main()
