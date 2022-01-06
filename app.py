#!/usr/bin/env python3

import numpy as np
import math
from os import system
import matplotlib.pyplot as plt
import random


#all_data
experience = [ 5,  10,  15,    5,  10, 15,    2,  4,   6   ] #years
projects   = [ 50, 100, 150,   25, 50, 75,    50, 100, 150 ] #number
earn       = [ 60, 70,  80,    30, 35, 40,    50, 60, 70   ] #k$/year


#data for testing
#experience = [ 5,   10  ] #years
#projects   = [ 100, 200 ] #number
#earn       = [ 120, 150 ] #k$/year



class Neuron:
    def __init__( self, w = 0, v = 0, b = 0, c = 0 ):
        self.w = w
        self.v = v
        self.b = b
        self.c = c
    
    def forward( self, X, Y ):
        return [ ( self.w * x + self.b ) * (self.v * y + self.c ) for x, y in zip( X, Y )  ] 


def error( Y, Y_pred ):
    return [ ( y - y_pred ) for y, y_pred in zip( Y, Y_pred )  ]


def loss( E ):                                     
    return sum( [ e * e for e in E ] ) / len( E )   #MSE

def accuracy( E ):
    return sum( [ abs( e ) for e in E ] ) / len( E )


w = 0.08
b = 1
v = 0.08
c = 1

def rough_approximation( X, Y, Z ):
    L = []
    
    w = np.random.normal()
    b = np.random.normal()
    v = np.random.normal()
    c = np.random.normal()
    
    for i in range( 20 ):
        n = Neuron( w, b, v, c )                            
        Y_predicted = n.forward( X, Y )
        E = error( Z, Y_predicted )
        L.append( loss( E ))
    return L





system( "clear" )

print( rough_approximation( experience, projects, earn ))
