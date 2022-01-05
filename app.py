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

plt.plot( experience, earn, color = "green", linestyle="solid", linewidth = 1, marker = "x" )
plt.show()

#data for testing
#experience = [ 5,   10  ] #years
#projects   = [ 100, 200 ] #number
#earn       = [ 120, 150 ] #k$/year



class Neuron:
    def __init__( self, w = 0, b = 0, a = 0 ):
        self.w = w
        self.b = b
        self.a = a
    
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
