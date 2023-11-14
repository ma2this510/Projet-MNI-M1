# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:18:41 2023

@author: tradu
"""
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(40)

class OSCI:
    def __init__(self,N):
        self.N=N
        self.ang=np.random.uniform(-np.pi,np.pi,N)
        
    def graph (self):
        cercle=plt.Circle((0,0),1,color='r',fill=False)
        plt.gca().add_patch(cercle)
        plt.scatter(np.cos(self.ang),np.sin(self.ang),color='b')
        ordre=(1/self.N)*np.sum(np.exp(1j*self.ang))
        plt.scatter(np.real(ordre),np.imag(ordre),color='g')
        plt.axis('equal')
        plt.grid()
        plt.savefig(f'kura1_{self.N}.pdf')
        plt.show()
        
        
blup=OSCI(10)
blup.graph()
