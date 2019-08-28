# Importing package tofu.geom
import tofu as tf
from tofu import __version__
import tofu.defaults as tfd
import tofu.pathfile as tfpf
import tofu.utils as tfu
import tofu.geom as tfg
import numpy as np

dmeths = ['rel', 'abs']
qmeths = ['simps', 'romb', 'sum']
list_res = [0.25, np.r_[0.2, 0.5]]
DL = np.array([[1.,10.],[2.,20.]])

for dL in list_res:
    for dm in dmeths:
        for qm in qmeths:
            print("============ for: ", dm, " -", qm, "=======")
            print("================= dl", dL," ================")
            out = tfg._GG.LOS_get_sample(2, dL, DL, dmethod=dm, method=qm)
            k = out[0]
            lind = out[2]
            print(" k1 =", k[:lind[0]])
            print(" k2 =", k[lind[0]:])
            print("lind =", lind)
            assert np.all(k[:lind[0]] >= DL[0][0])
            assert np.all(k[:lind[0]] <= DL[1][0])
            assert np.all(k[lind[0]:] >= DL[0][1])
            assert np.all(k[lind[0]:] <= DL[1][1])
            print("================= OK =======================")
