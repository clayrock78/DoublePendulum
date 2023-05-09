import cv2
import numpy as np
from scipy.integrate import odeint
import numexpr as ne 
from time import perf_counter

def hsv_to_rgb(h,s,v):
    # works with numpy arrays
    r = np.zeros(h.shape)
    g = np.zeros(h.shape)
    b = np.zeros(h.shape)

    h_i = (h*6).astype(int)
    f = h*6 - h_i
    p = v * (1-s)
    q = v * (1-f*s)
    t = v * (1-(1-f)*s)
    r[h_i==0] = v[h_i==0]
    g[h_i==0] = t[h_i==0]
    b[h_i==0] = p[h_i==0]
    r[h_i==1] = q[h_i==1]
    g[h_i==1] = v[h_i==1]
    b[h_i==1] = p[h_i==1]
    r[h_i==2] = p[h_i==2]
    g[h_i==2] = v[h_i==2]
    b[h_i==2] = t[h_i==2]
    r[h_i==3] = p[h_i==3]
    g[h_i==3] = q[h_i==3]
    b[h_i==3] = v[h_i==3]
    r[h_i==4] = t[h_i==4]
    g[h_i==4] = p[h_i==4]
    b[h_i==4] = v[h_i==4]
    r[h_i==5] = v[h_i==5]
    g[h_i==5] = p[h_i==5]
    b[h_i==5] = q[h_i==5]
    return np.array([r*255, g*255, b*255]).astype(int)



# ----------------------- PENDULUM CLASS DEFINITON -----------------------

def RK4(t, y, h, f):  
    
    #Runge Kutta standard calculations
    k1 = f(t, y)
    k2 = f(t + h/2, y + h/2 * k1)
    k3 = f(t + h/2, y + h/2 * k2)
    k4 = f(t + h, y + h * k3)    

    return ne.evaluate('1/6*(k1 + 2 * k2 + 2 * k3 + k4)')

    
def RHS(t, y):
    # np array of shape (?,4) to 4 numpy arrays of shape (?)
    theta_1, theta_2, w1, w2 = y
    ################################
    #Critical physical Parameters
    g = 15
    l1 = 1
    l2 = 1
    m1 = 1
    m2 = 1
    delta_theta = theta_1 - theta_2
    ################################
    
    #Writing the system of ODES.
    f0 = w1
    f1 = w2


    f2 = '(m2*l1*w1**2*sin(2*delta_theta) + 2*m2*l2*w2**2*sin(delta_theta) + 2*g*m2*cos(theta_2)*sin(delta_theta) + 2*g*m1*sin(theta_1))/(-2*l1*(m1 + m2*sin(delta_theta)**2))'
    f2 = ne.evaluate(f2)
    
    f3 = '(m2*l2*w2**2*sin(2*delta_theta) + 2*(m1 + m2)*l1*w1**2*sin(delta_theta) + 2*g*(m1 + m2)*cos(theta_1)*sin(delta_theta))/(2*l2*(m1 + m2*sin(delta_theta)**2))'
    f3 = ne.evaluate(f3)
    
    return np.array([f0, f1, f2, f3])



def pos_as_color(p1,p2):
    # represents 2 positions as color
    h = ne.evaluate('p1 / width * .7 + .3')
    s = ne.evaluate('p2 / width / 2 + 0.5')
    v = ne.evaluate('p2 / width / 2 + 0.5')
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

def angles_as_pos(a1, a2):
    # represents 2 angles as position
    x1 = ne.evaluate('1 * sin(a1)')
    y1 = ne.evaluate('-1 * cos(a1)')
    x2 = ne.evaluate('x1 + 1 * sin(a2)')
    y2 = ne.evaluate('y1 - 1 * cos(a2)')
    x2 = ne.evaluate('x2 * width / 5 + width / 2')
    y2 = ne.evaluate('y2 * width / 5 + width / 2')
    return x2, y2

if __name__ == "__main__":
    width = 500
    screen = np.zeros((width,width,3),dtype=np.uint8)
    frames = []

    # time interval
    t = 0 
    T = 5
    n = 300
    # initial conditions
    # y is a 4x(width**2) array of initial conditions these initial conditions represent the initial angles and angular velocities of the pendulums
    a1 = np.linspace(0, 2*np.pi, width)
    a2 = np.linspace(0, 2*np.pi, width)
    y = np.meshgrid(a1, a2)
    y = np.array([y[0].flatten(), y[1].flatten(), np.zeros(width**2), np.zeros(width**2)])
    
    #Step size
    h  = (T - t)/n
    
    height, width, layers = (width,width,3)
    video = cv2.VideoWriter('video_np.mp4', 0, 60, (width,height))
    while t < T:
        screen = pos_as_color(*angles_as_pos(y[0],y[1]))
        video.write(screen.T.copy().reshape((height,width,3)).astype(np.uint8))
        y = y + h * RK4(t, y, h, RHS)
        t = t + h
        print(f"{(t)/T*100:.2f} %")
    cv2.destroyAllWindows()
    video.release()
