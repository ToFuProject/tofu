# -*- coding: utf-8 -*-
"""
Created on Sept 30, 2015 13:07

@author: didiervezinet

This module provides the basic tools for computing quadrature points for a TFM.BF2D object and using them to compute an integral
"""

__author__ = 'Didier Vezinet (didier.vezinet@ipp.mpg.de)'
__version__ = 'V0'
__date__ = "$Sept 30, 2015 13:07$"
__copyright__ = ' '
__license__ = 'Eurofusion'



import numpy as np
from math import cos,pi
import matplotlib.pyplot as plt
import ToFu_Mesh as TFM





def _getQuadPtsWght(BF2):







def _getQuadPtsWght(k, Type='Uniform'):
    assert type(k) is int, "Arg k must be a int !"
    assert type(Type) is str and Type in ['uniform','Gauss-Lobatto','Gauss-Legendre'], "Arg Type must be a str in ['uniform','Gauss-Lobatto','Gauss-Legendre'] !"

    if Type=='uniform':
        xg = np.linspace(-1., 1., k+1)
        w  = np.ones(k+1)

    elif Type=='Gauss-Lobatto':
        beta = .5 / np.sqrt(1-(2 * np.arange(1., k + 1)) ** (-2)) #3-term recurrence coeffs
        beta[-1] = np.sqrt((k / (2 * k-1.)))
        T = np.diag(beta, 1) + np.diag(beta, -1) # jacobi matrix
        D, V = np.linalg.eig(T) # eigenvalue decomposition
        xg = np.real(D)
        i = xg.argsort()
        xg.sort() # nodes (= Legendres points)
        w = 2 * (V[0, :]) ** 2; # weights

    elif Type=='Gauss-Legendre':
        m = ordergl + 1

        def legendre(t,m):
            p0 = 1.0; p1 = t
            for k in range(1,m):
                p = ((2.0*k + 1.0)*t*p1 - k*p0)/(1.0 + k )
                p0 = p1; p1 = p
            dp = m*(p0 - t*p1)/(1.0 - t**2)
            return p1,dp

        xg, w = np.zeros(m), np.zeros(m)
        nRoots = (m + 1)/2          # Number of non-neg. roots
        for i in range(0,nRoots):
            t = cos(pi*(i + 0.75)/(m + 0.5))  # Approx. root
            for j in range(0,30):
                p,dp = legendre(t,m)          # Newton-Raphson
                dt = -p/dp
                t = t + dt        # method
                if abs(dt) < tol:
                    xg[i] = t
                    xg[m-i-1] = -t
                    w[i] = 2.0/(1.0 - t**2)/(dp**2) # Eq.(6.25)
                    w[m-i-1] = w[i]
                    break
        return xg, w






