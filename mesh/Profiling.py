#!/usr/bin/env python
# encoding: utf-8
# filename: profile.py

import pstats, cProfile

import _bsplines_cy as bsp_cy



#cProfile.runctx("bsp_cy.Calc_1D_IntOp(Method='quad')", globals(), locals(), "Profile.prof")
cProfile.runctx("bsp_cy.Calc_1D_LinIntOp(Method='')", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()




