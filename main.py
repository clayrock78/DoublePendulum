import numpy as np
from scipy.integrate import odeint
import math
import threading


# ----------------------- CONSTANTS AND CALCULATIONS -----------------------
# Pendulum rod lengths (m), bob masses (kg).
L1, L2 = 1, 1
m1, m2 = 1, 1
# The gravitational acceleration (m.s-2).
g = 9.81

def deriv(y, t, L1, L2, m1, m2):
    #print("deriving")
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, z1, theta2, z2 = y

    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    theta1dot = z1
    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
             (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
    theta2dot = z2
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
             m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
    return theta1dot, z1dot, theta2dot, z2dot

def calc_E(y):
    """Return the total energy of the system."""

    th1, th1d, th2, th2d = y.T
    V = -(m1+m2)*L1*g*np.cos(th1) - m2*L2*g*np.cos(th2)
    T = 0.5*m1*(L1*th1d)**2 + 0.5*m2*((L1*th1d)**2 + (L2*th2d)**2 +
            2*L1*L2*th1d*th2d*np.cos(th1-th2))
    return T + V

def hsv_to_rgb(h, s, v):
    # h = 0 to 1
    # s = 0 to 1
    # v = 0 to 1
    # returns r, g, b in 0 to 255
    r = 0
    g = 0
    b = 0

    h_i = int(h*6)
    f = h*6 - h_i
    p = v * (1-s)
    q = v * (1-f*s)
    t = v * (1-(1-f)*s)
    if h_i == 0: r, g, b = v, t, p
    if h_i == 1: r, g, b = q, v, p
    if h_i == 2: r, g, b = p, v, t
    if h_i == 3: r, g, b = p, q, v
    if h_i == 4: r, g, b = t, p, v
    if h_i == 5: r, g, b = v, p, q
    return int(r*255), int(g*255), int(b*255)


# ----------------------- PENDULUM CLASS DEFINITON -----------------------
class pendulum:
    def __init__(self, x_init, y_init, talker=False):
        self.talker = talker
        self.x_init = x_init
        self.y_init = y_init
        self.calculated = False

    def calc(self, width = 1000):
        theta1 = self.x_init / width * 360
        theta2 = self.y_init / width * 360
        # Maximum time, time point spacings and the time grid (all in s).
        tmax, dt = 10, 0.01
        self.t = np.arange(0, tmax+dt, dt)
        # Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
        self.y0 = np.array([math.radians(theta1), 0, math.radians(theta2), 0])
        # Do the numerical integration of the equations of motion
        y = odeint(deriv, self.y0, self.t, args=(L1, L2, m1, m2))

        # Unpack z and theta as a function of time
        theta1, theta2 = y[:,0], y[:,2]

        # Convert to Cartesian coordinates of the two bob positions.
        self.x1 = L1 * np.sin(theta1)
        self.y1 = -L1 * np.cos(theta1)
        self.x2 = self.x1 + L2 * np.sin(theta2)
        self.y2 = self.y1 - L2 * np.cos(theta2)

        if self.talker:
            print(self.x_init)

        return self.x2, self.y2, self.x_init, self.y_init


def pos_as_color(p1,p2):
    # represents 2 positions as color
    h = p1 / width
    s = p2 / width / 2 + 0.5
    v = p2 / width / 2 + 0.5
    return hsv_to_rgb(h,s,v)
    

def make_plot(x2, y2, x_init, y_init):
    # the position of the pendulum at time i is calculated
    x = x2[i] * width / 5 + width / 2
    y = y2[i] * width / 5 + width / 2
    #print(x,y)
    # the position is converted to a color
    color = pos_as_color(x,y)
    # the pixel at the position on the screen gets colored
    screen[int(x_init),int(y_init)] = color

if __name__ == "__main__":
    width = 1000
    screen = np.zeros((width,width,3),dtype=np.uint8)
    frames = []

    pends = []

    # ----------------------- INITIALIZATION -----------------------
    # create pendulum objects
    for x in range(width):
        for y in range(width):
            pends.append(pendulum(x,y,y==0))
        
    #for pend in pends:
    #    pend.calc()

    print("starting pool")
    # create thread pool
    from multiprocessing import Pool

    with Pool(4) as p:
        stuff = p.map(pendulum.calc, pends)
    print("done")


    # ----------------------- MAIN LOOP -----------------------
    i = 0
    running = True
    while running:
        try:
            for s in stuff:
                make_plot(*s)
            frames.append(screen.copy())
            i+= 1
        except:
            running = False

    # save frames as video
    import cv2
    height, width, layers = frames[0].shape
    video = cv2.VideoWriter('video_test.mp4', 0, 60, (width,height))
    for frame in frames:
        video.write(frame)
    cv2.destroyAllWindows()
    video.release()
