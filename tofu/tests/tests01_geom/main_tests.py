from tests01_GG import *
import datetime as dtm


t0 = dtm.datetime.now()
test01_CoordShift()
t1 = dtm.datetime.now()
dt01 = (t1-t0).total_seconds()
print("time test1 CoordShift = ", dt01)

t0 = dtm.datetime.now()
test02_Poly_CLockOrder()
t1 = dtm.datetime.now()
dt01 = (t1-t0).total_seconds()
print("time test2 Poly_ClockOrder = ", dt01)

t0 = dtm.datetime.now()
test03_Poly_VolAngTor()
t1 = dtm.datetime.now()
dt01 = (t1-t0).total_seconds()
print("time test3 Poly_VolAngTor = ", dt01)

# Ves ................................

t0 = dtm.datetime.now()
test04_Ves_isInside()
t1 = dtm.datetime.now()
dt01 = (t1-t0).total_seconds()
print("time test4 Ves_isInside = ", dt01)

t0 = dtm.datetime.now()
test05_Ves_mesh_dlfromL()
t1 = dtm.datetime.now()
dt01 = (t1-t0).total_seconds()
print("time test05 Ves_mesh_dlfromL  = ", dt01)

t0 = dtm.datetime.now()
test06_Ves_Smesh_Cross()
t1 = dtm.datetime.now()
dt01 = (t1-t0).total_seconds()
print("time test06 Smesh Cross  = ", dt01)

t0 = dtm.datetime.now()
test07_Ves_Vmesh_Tor()
t1 = dtm.datetime.now()
dt01 = (t1-t0).total_seconds()
print("time test07 Ves_Vmesh_Tor  = ", dt01)

t0 = dtm.datetime.now()
test08_Ves_Vmesh_Lin()
t1 = dtm.datetime.now()
dt01 = (t1-t0).total_seconds()
print("time test08_Ves_Vmesh_Lin  = ", dt01)

t0 = dtm.datetime.now()
test09_Ves_Smesh_Tor()
t1 = dtm.datetime.now()
dt01 = (t1-t0).total_seconds()
print("time test09_Ves_Smesh_Tor  = ", dt01)

t0 = dtm.datetime.now()
test10_Ves_Smesh_Tor_PhiMinMax()
t1 = dtm.datetime.now()
dt01 = (t1-t0).total_seconds()
print("time test10_Ves_Smesh_Tor_PhiMinMax  = ", dt01)

t0 = dtm.datetime.now()
test11_Ves_Smesh_TorStruct()
t1 = dtm.datetime.now()
dt01 = (t1-t0).total_seconds()
print("time test11_Ves_Smesh_TorStruct  = ", dt01)

t0 = dtm.datetime.now()
test12_Ves_Smesh_Lin()
t1 = dtm.datetime.now()
dt01 = (t1-t0).total_seconds()
print("time test12_Ves_Smesh_Lin  = ", dt01)


# LOS......................
t0 = dtm.datetime.now()
test13_LOS_PInOut()
t1 = dtm.datetime.now()
dt01 = (t1-t0).total_seconds()
print("time test13_LOS_PInOut  = ", dt01)


print(".......all passed......")
