# Nose-specific
from nose import with_setup # optional

# ToFu-specific
# Plugin-specific
import tofu.plugins.West.Ves as tfWV

#######################################################
#
#     Setup and Teardown
#
#######################################################

def setup_module(module):
    print ("") # this is to get a newline after the dots
    #print ("setup_module before anything in this file")

def teardown_module(module):
    #os.remove(VesTor.Id.SavePath + VesTor.Id.SaveName + '.npz')
    #os.remove(VesLin.Id.SavePath + VesLin.Id.SaveName + '.npz')
    #print ("teardown_module after everything in this file")
    #print ("") # this is to get a newline
    pass


#def my_setup_function():
#    print ("my_setup_function")

#def my_teardown_function():
#    print ("my_teardown_function")

#@with_setup(my_setup_function, my_teardown_function)
#def test_numbers_3_4():
#    print 'test_numbers_3_4  <============================ actual test code'
#    assert multiply(3,4) == 12

#@with_setup(my_setup_function, my_teardown_function)
#def test_strings_a_3():
#    print 'test_strings_a_3  <============================ actual test code'
#    assert multiply('a',3) == 'aaa'






#######################################################
#
#     Testing
#
#######################################################



"""
######################################################
######################################################
#               Commons
######################################################
######################################################
"""

def test01_create():

    # Create from Poly
    thet = np.linspace(0.,2.*np.pi,100)
    Poly = np.array([2.5+1.*np.cos(thet), 0.+1.*np.sin(thet)])
    V0 = tfWV.create_Ves('test01-V0', src=Poly, save=False)    
    S = tfWV.create_Struct('test01-V0', src=Poly, save=False)

    # Create from PathFileExt
    V1 = tfWV.create_Struct('test01-V1', src='/Home/DV226270/ToFu_All/tofu_git/tofu/tofu/plugins/WEST/Ves/Inputs/WEST_Ves_light.txt', save=False)
    S1 = tfWV.create_Struct('test01-V1', src='/Home/DV226270/ToFu_All/tofu_git/tofu/tofu/plugins/WEST/Ves/Inputs/WEST_Struct_OuterBumper_light.txt', save=False)
    # Create from dict (imas)
    V2 = tfWV.create_Ves('test01-V1', src={'from':'imas', shot=0, }, save=False)
    S2 = tfWV.create_Struct('test01-V1', src={'from':'imas', shot=0, }, save=False)



def test02_loadplot():
    V = tfWV.load_Ves()
    S = tfWV.load_STruct()
     
    Lax = V.plot()
    Lax = S.plot(Lax=Lax) 
    
    VmS = V.get_meshS()
    VmV = V.get_meshV()
    SmS = S.get_meshS()

    

    






















