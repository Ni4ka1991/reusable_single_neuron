#!/usr/bin/env python3

import numpy as np
import math
from os import system
import matplotlib.pyplot as plt
import random
#from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax  = plt.axes( projection = '3d' )
#1
#y = 10 * x

experience_1 = [ 5,  10,  15  ] 
projects_1   = [ 50, 100, 150 ]
earn_1       = [ 60, 70,  80  ]

#ax.plot3D( experience_1, projects_1, earn_1, 'green', label ='parametric curve' )
#ax.set_title( 'firs data set' )
#plt.show()


#2
experience_2 = [ 5,  10, 15 ] #years
projects_2   = [ 25, 50, 75 ] #number
earn_2       = [ 30, 35, 40 ] #k$/year

#ax.plot3D( experience_2, projects_2, earn_2, 'red', label ='parametric curve' )
#ax.set_title( 'Two data sets' )
#plt.show()

#3
experience_3 = [ 2,  4,   6   ] #years
projects_3   = [ 50, 100, 150 ] #number
earn_3       = [ 50, 60, 70   ] #k$/year

#ax.plot3D( experience_3, projects_3, earn_3, 'blue', label ='parametric curve' )
#ax.set_title( 'Three data sets' )
#plt.show()

#all_data
experience = [ 5,  10,  15,    5,  10, 15,    2,  4,   6   ] #years
projects   = [ 50, 100, 150,   25, 50, 75,    50, 100, 150 ] #number
earn       = [ 60, 70,  80,    30, 35, 40,    50, 60, 70   ] #k$/year

ax.plot3D( experience, projects, earn, 'black', label ='parametric curve' )
ax.set_title( 'All data sets' )
plt.show()


#z = np.linspace( 0, 1, 100 )
#x = z * np.sin( 25 * z )
#y = z * np.cos( 25 * z )


class Neuron:
    def __init__( self, w = 0, b = 0 ):
        self.w = w
        self.b = b
    
    def forward( self, X ):
        return [ ( self.w * x + self.b ) for x in X  ] 


def error( Y, Y_pred ):
    return [ ( y - y_pred ) for y, y_pred in zip( Y, Y_pred )  ]


def loss( E ):                                     
    return sum( [ e * e for e in E ] ) / len( E )   #MSE

def accuracy( E ):
    return sum( [ abs( e ) for e in E ] ) / len( E )


n = Neuron( 3, 5 )               #neuron initialization with w = 3 and b = 5
Y_predicted = n.forward( X )     #calculation for all X, Y_predicted ( w = 3, b = 5 )

E = error( Y, Y_predicted )

system( "clear" )
print( E )
