#!/usr/bin/env python3

import numpy as np
import math
from os import system
import matplotlib.pyplot as plt
import random
#from mpl_toolkits.mplot3d import Axes3D

#fig = plt.figure()
#ax  = plt.axes( projection = '3d' )

#all_data
experience = [ 5,  10,  15,    5,  10, 15,    2,  4,   6   ] #years
projects   = [ 50, 100, 150,   25, 50, 75,    50, 100, 150 ] #number
earn       = [ 60, 70,  80,    30, 35, 40,    50, 60, 70   ] #k$/year

#ax.plot3D( experience, projects, earn, 'black', label ='parametric curve' )
#ax.set_title( 'All data sets' )

#plt.plot( experience, earn, color = "green", linestyle="solid", linewidth = 1, marker = "x" )
#plt.show()

#data for testing
#experience = [ 5,   10  ] #years
#projects   = [ 100, 200 ] #number
#earn       = [ 120, 150 ] #k$/year


    
def search_dependencies( X, Y, Z ):
        return [ ( x * y ) / z for x, y, z in zip( X, Y, Z ) ] 


coef = search_dependencies( experience, projects, earn )
coef_1 = coef[:3]
coef_2 = coef[3:6]
coef_3 = coef[6:]


#plt.plot( experience, earn, color = "green", linestyle="solid", linewidth = 1, marker = "x" )
#plt.show()

system( "clear" )
print( f"coef = {coef}" )
print( f"coef_1 = {coef_1}" )
print( f"coef_2 = {coef_2}" )
print( f"coef_3 = {coef_3}" )

plt.plot( coef_1, color = "green", linestyle="solid", linewidth = 1, marker = "x" )
plt.plot( coef_2, color = "blue",  linestyle="solid", linewidth = 1, marker = "x" )
plt.plot( coef_3, color = "red",   linestyle="solid", linewidth = 1, marker = "x" )
plt.show()
