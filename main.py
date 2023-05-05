import pygame
import sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math

pygame.init()

screen = pygame.display.set_mode([500, 500])

# ----------------------- CONSTANTS AND CALCULATIONS -----------------------
# Pendulum rod lengths (m), bob masses (kg).
L1, L2 = 1, 1
m1, m2 = 1, 1
# The gravitational acceleration (m.s-2).
g = 9.81

def deriv(y, t, L1, L2, m1, m2):
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
    def __init__(self, theta1, theta2):

        # Maximum time, time point spacings and the time grid (all in s).
        tmax, dt = 5, 0.01
        t = np.arange(0, tmax+dt, dt)
        # Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
        y0 = np.array([math.radians(theta1), 0, math.radians(theta2), 0])
        # theta angles represent positions on the screen (of all possible positions)
        self.x_init = theta1 / 360 * 500
        self.y_init = theta2 / 360 * 500

        # Do the numerical integration of the equations of motion
        y = odeint(deriv, y0, t, args=(L1, L2, m1, m2))

        # Unpack z and theta as a function of time
        theta1, theta2 = y[:,0], y[:,2]

        # Convert to Cartesian coordinates of the two bob positions.
        self.x1 = L1 * np.sin(theta1)
        self.y1 = -L1 * np.cos(theta1)
        self.x2 = self.x1 + L2 * np.sin(theta2)
        self.y2 = self.y1 - L2 * np.cos(theta2)


    def pos_as_color(self,p1,p2):
        # represents 2 positions as color
        h = p1 / 500
        s = p2 / 500
        v = 1
        return hsv_to_rgb(h,s,v)
    

    def make_plot(self, screen, i):
        x = self.x2[i]*100+250
        y = -self.y2[i]*100+250
        color = self.pos_as_color(x,y)
        # the pixel at the position on the screen gets colored
        pygame.draw.circle(screen, (*color,255), (self.x_init, self.y_init), 7)

# Make an image every di time points, corresponding to a frame rate of fps
# frames per second.
# Frame rate, s-1
p = []
for x in range(1,50):
    for y in range(1,50):
        p.append(pendulum(x/50*360,y/50*360))
    print(x)

i = 0
clock = pygame.time.Clock()
running = True
fade_layer = pygame.Surface((500,500), pygame.SRCALPHA)
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
    for pend in p:
        pend.make_plot(screen, i)
    # fade the screen layer
    fade_layer.fill((0,0,0,1))
    screen.blit(fade_layer, (0,0))



    pygame.display.flip()
    clock.tick(1000)
    i += 1

pygame.quit()
