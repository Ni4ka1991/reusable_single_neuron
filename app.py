#!/usr/bin/env python3

import numpy as np
import math
from os import system
import matplotlib.pyplot as plt
import random
#from mpl_toolkits.mplot3d import Axes3D

#1
#y = 10 * x

experience_1 = [ 5,  10,  15  ] 
projects_1   = [ 50, 100, 150 ]
earn_1       = [ 60, 70,  80  ]

plt.plot( experience_1, projects_1, color = "green", linestyle="solid", linewidth = 1, marker = "x" )
plt.plot( experience_1, earn_1,     color = "red",   linestyle="solid", linewidth = 1, marker = "x" )
plt.plot( projects_1,   earn_1,     color = "blue",  linestyle="solid", linewidth = 1, marker = "x" )
plt.show()


#2
experience = [ 5,  10, 15 ] #years
projects   = [ 25, 50, 75 ] #number
earn       = [ 30, 35, 40 ] #k$/year

#3
experience = [ 2,  4,   6   ] #years
projects   = [ 50, 100, 150 ] #number
earn       = [ 50, 60, 70   ] #k$/year




#X = [ 5, 10, 15, 20 ]
#Y = [ 5, 10, 10, 15 ]

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
