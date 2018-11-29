# -*- coding: utf-8 -*-
#! /usr/bin/python
"""
The geometry module of tofu

Provides classes to model the 3D geometry of:
* the vacuum vessel and structural elements
* LOS
* apertures and detectors
"""

from tofu.geom._core import PlasmaDomain, Ves, PFC, CoilPF, CoilCS, Config, Rays, LOSCam1D, LOSCam2D

__all__ = ['_GG', '_comp', '_plot', '_def']
