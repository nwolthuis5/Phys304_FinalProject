import numpy as np
import matplotlib.pyplot as plt

G = 6.674*10**(-11) #m^3/kg s^2
Msun = 1.989*10**30 #kg
AU = 1.496*10**11 #m
#1 is ACB, 2 is ACA, 3 is Barnard's
M1 = 9.07*Msun
M2 = 1.10*Msun
M3 = 0.144*Msun

#dist between 1 and 2 is 23 AU

array = [x1, x2, x3, y1, y2, y3, x1p, x2p, x3p, y1p, y2p, y3p]

r12 = ((x2-x1)**2+(y2-y1)**2)**(1/2)
r23 = ((x3-x2)**2+(y3-y2)**2)**(1/2)
r13 = ((x3-x1)**2+(y3-y1)**2)**(1/2)

#we define a function to represent our system of equations
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
    
#Runge-kutta method
for t in tpoints:
    #we append the values to the lists
    driventhetapoints.append(r2[0])
    drivenomegapoints.append(r2[1])
    #then we run through all our k's
    k1 = h*driven(r2,t)
    k2 = h*driven(r2+0.5*k1,t+0.5*h)
    k3 = h*driven(r2+0.5*k2,t+0.5*h)
    k4 = h*driven(r2+k3,t+h)
    #and add it on to the previous value of r2
    r2 += (k1 + 2*k2 + 2*k3 + k4)/6
