#importing packages
import numpy as np
import matplotlib.pyplot as plt
#this is the package we use to do the animations
from matplotlib.animation import FuncAnimation

#defining constants
#we used units of AU, solar masses, and days
G = 3*10**(-4) #AU^3/Solar mass days^2
Msun = 1
AU = 1


#user inputs for the mass ratios of our three bodies
#we used the convention that M1 was largest and M3 smallest
M1 = float(input("please provide a real value for M1 in units of solar masses: "))
M2 = float(input("please provide a real value for M2 in units of solar masses: "))
M3 = float(input("please provide a real value for M3 in units of solar masses: "))

#this is the mass equivalent we decided to use for our initial conditions
#it's based off the reduced mass for two bodies
#there's an invisible constant of 1 mass^(-1)
M = (M1 * M2 * M3)/(M1 + M2 + M3)

#initial positions
#the bodies are placed at the vertices of an equilateral triangle
#with side length 2* sqrt(3)
x1 = -np.sqrt(3)
y1 = -1
x2 = 0
y2 = 2
x3 = np.sqrt(3)
y3 = -1

#we decided to make a velocity based on the orbital velocity of a satellite
#it will scale with the masses
#the distance from the origin to each point is 2
orbital_R = 2

#we multiply this starting velocity by trig stuff
#to get the x and y components of it
factor= np.sqrt(2*G*M/orbital_R)

#here is where we get the x and y components
#p stands for prime, as in the derivative
x1p = -1/2 * factor
y1p = np.sqrt(3)/2 * factor
x2p = 1 * factor
y2p = 0 * factor

#we need the net momentum to be zero
#so we define the first two momentums here
x1_mom = M1*x1p
y1_mom = M1*y1p

x2_mom = M2*x2p
y2_mom = M2*y2p

#the third one is defined such that the total momentum
#is zero
x3_mom = -(x1_mom + x2_mom)
x3p = x3_mom/M3

y3_mom = -(y1_mom + y2_mom)
y3p = y3_mom/M3


#setting up an array with starting values
#for our runge kutta
array = np.array([x1, y1, x1p, y1p,
                  x2, y2, x2p, y2p,
                  x3, y3, x3p, y3p], float)

#distances between each pair of bodies
r12 = ((x2-x1)**2+(y2-y1)**2)**(1/2)
r23 = ((x3-x2)**2+(y3-y2)**2)**(1/2)
r13 = ((x3-x1)**2+(y3-y1)**2)**(1/2)

#we define a function to represent our system of equations
#we have to have 12 variables, four for each body
def f(r,t):
    #setting vars for M1
    u1 = array[0] #x1
    u2 = array[1] #y1
    u3 = array[2] #x1p
    u4 = array[3] #y1p

    #setting vars for M2
    u5 = array[4] #x2
    u6 = array[5] #y2
    u7 = array[6] #x2p
    u8 = array[7] #y2p

    #setting vars for
    u9 = array[8] #x3
    u10 = array[9] #y3
    u11 = array[10] #x3p
    u12 = array[11] #y3p

#this is for M1
    u1p = u3 #deriv of pos is equal to velocity, x version
    u2p = u4 #deriv of pos is equal to velocity, y version
    u3p = G*M2/(r12**3)*(u5-u1)+G*M3/(r13**3)*(u9-u1) #x acceleration
    u4p = G*M2/(r12**3)*(u6-u2)+G*M3/(r13**3)*(u10-u2) #y acceleration

#this is for M2
    u5p = u7 #deriv of pos is equal to velocity
    u6p = u8
    u7p = G*M1/(r12**3)*(u1-u5)+G*M3/(r23**3)*(u9-u5)
    u8p = G*M1/(r12**3)*(u2-u6)+G*M3/(r23**3)*(u10-u6)

#this is for M3
    u9p = u11 #deriv of pos is equal to velocity
    u10p = u12
    u11p = G*M1/(r13**3)*(u1-u9)+G*M2/(r23**3)*(u5-u9)
    u12p = G*M1/(r13**3)*(u2-u10)+G*M2/(r23**3)*(u6-u10)

    #then we have equations which are our desired outputs
    #we can basically treat derivatives as simple variables

    return np.array([u1p,u2p,u3p,u4p,u5p,u6p,u7p,u8p,u9p,u10p,u11p,u12p],float)

#set up our time limits over which to look at the system
a = 0.0
b = 10**4

#this is the number of points we check at
N = 100000
#and the bin size for our time step
h = (b-a)/N

#set a time array and empty lists for x and y
tpoints = np.arange(a,b,h)

#making empty lists for runge kutta :(
#we need a separate list for each velocity and position
l_x1, l_y1, l_vx1, l_vy1, l_x2, l_y2, l_vx2, l_vy2, l_x3, l_y3, l_vx3, l_vy3 = [], [], [], [], [], [], [], [], [], [], [], []

#Runge-kutta method
for t in tpoints:
    #we append the values to the lists
    l_x1.append(array[0])
    l_y1.append(array[1])
    l_vx1.append(array[2])
    l_vy1.append(array[3])

    l_x2.append(array[4])
    l_y2.append(array[5])
    l_vx2.append(array[6])
    l_vy2.append(array[7])

    l_x3.append(array[8])
    l_y3.append(array[9])
    l_vx3.append(array[10])
    l_vy3.append(array[11])

    #then we run through all our k's
    k1 = h*f(array,t)
    k2 = h*f(array+0.5*k1,t+0.5*h)
    k3 = h*f(array+0.5*k2,t+0.5*h)
    k4 = h*f(array+k3,t+h)
    #and add it on to the previous value of array
    array += (k1 + 2*k2 + 2*k3 + k4)/6



#determining maximum values for our axes
#we want to make our axes go to the highest value that the system reaches
#and we want it to be square to avoid warping the image
#we make a list of all x's and find the largest one, then peel that off
xmaxlist = np.concatenate([l_x1,l_x2,l_x3])
xmaxlist = np.sort(np.abs(xmaxlist))
x_max = xmaxlist[-1]

#similarly for y
ymaxlist = np.concatenate([l_y1,l_y2,l_y3])
ymaxlist = np.sort(np.abs(ymaxlist))
y_max = ymaxlist[-1]

#we choose which one is bigger and that defines our axis limits
maxlist = [y_max, x_max]
maxlist = np.sort(np.abs(maxlist))
maxval = maxlist[-1]

#code from when we were making static plots

# definitions for the axes
#left, width = 0.1, 0.65
#bottom, height = 0.1, 0.65
#spacing = 0.01
#rect_scatter = [left, bottom, width, height]
# start with a rectangular Figure
#plt.figure(figsize=(8, 8))
#ax_scatter = plt.axes(rect_scatter)
#ax_scatter.tick_params(direction='in', top=True, right=True)
# the scatter plot:
#ax_scatter.scatter(l_x1, l_y1, color = 'k', alpha = .01)
#ax_scatter.scatter(l_x2, l_y2, color = 'b', alpha = .01)
#ax_scatter.scatter(l_x3, l_y3, color = 'orange', alpha = .01)
#ax_scatter.scatter(startpoints[0],startpoints[1],color= 'r')
#limits
#ax_scatter.set_xlim(-x_max, x_max)
#ax_scatter.set_ylim(-y_max, y_max)
#Formatting, Labels, & Legends
#plt.xlabel('spase')
#plt.ylabel('spase 2')
#plt.title('move')
#plt.show()

#empty lists for animation
x_data1, y_data1, x_data2, y_data2, x_data3, y_data3 = [],[],[],[],[],[]

#defining parts of figure to be animated
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False)

#choosing limits
ax.set_xlim(-maxval, maxval)
ax.set_ylim(-maxval,maxval)
#defining the lines to be animated
line1, = ax.plot(0,0)
line2, = ax.plot(0,0)
line3, = ax.plot(0,0)

#animation 1
#it will plot the entire set of points from the beginning til the current time
def animation_frame_1(i):
    x_data1.append(l_x1[i]) #appending each new step from the runge kutta
    y_data1.append(l_y1[i])

    line1.set_xdata(x_data1) #pairing it with the animated line
    line1.set_ydata(y_data1)
    line1.set_color('k') #make M1 black
    return line1,

#animation 2
def animation_frame_2(i):
    x_data2.append(l_x2[i]) #appending each new step from the runge kutta
    y_data2.append(l_y2[i])

    line2.set_xdata(x_data2) #pairing it with the animated line
    line2.set_ydata(y_data2)
    line2.set_color('b') #make M2 blue
    return line2,

#animation 3
def animation_frame_3(i):
    x_data3.append(l_x3[i]) #appending each new step from the runge kutta
    y_data3.append(l_y3[i])

    line3.set_xdata(x_data3) #pairing it with the animated line
    line3.set_ydata(y_data3)
    line3.set_color('orange') #make M3 orange
    return line3,

#number of frames is the same as the number of time points
frame_number =int(len(tpoints))

#actually putting the animations down
#func tells it which array to look at
#frames is the number of frames
#interval is the amount of time between frames in ms
animation1 = FuncAnimation(fig, func = animation_frame_1, frames=frame_number, interval=1)
animation2 = FuncAnimation(fig, func = animation_frame_2, frames=frame_number, interval=1)
animation3 = FuncAnimation(fig, func = animation_frame_3, frames=frame_number, interval=1)

#spacing for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]



#axis labels
plt.xlabel('X Distance from Center in AU')
plt.ylabel('Y Distance from Center in AU')



#tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()
