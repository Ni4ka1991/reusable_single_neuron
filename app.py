#!/usr/bin/env python3

import numpy as np
import math
from os import system
import matplotlib.pyplot as plt
import random


X = [ 5, 10, 15, 20 ]
Y = [ 5, 10, 10, 15 ]

class Neuron:
    def __init__( self, w = 0, b = 0 ):
        self.w = w
        self.b = b
    
    def forward( self, X ):
        return [ ( self.w * x + self.b ) for x in X  ] 


def error( Y, Y_pred ):
    return [ ( y - y_pred ) for y, y_pred in zip( Y, Y_pred )  ]


def loss( E ):
    return sum( [ e * e for e in E ] ) / len( E )

def accuracy( E ):
    return sum( [ abs( e ) for e in E ] ) / len( E )


n = Neuron( 3, 5 )               #neuron initialization with w = 3 and b = 5
Y_predicted = n.forward( X )     #calculation for all X, Y_predicted ( w = 3, b = 5 )

E = error( Y, Y_predicted )

system( "clear" )
print( E )
