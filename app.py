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


n = []
n = Neuron.forward( X )
system( "clear" )
print( n )
