# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 23:29:09 2019

@author: napra
"""

def veri_traj(traj):
    """This trajectory verifies the correct relation has been assigned 
    to all the clusters.
    
    Parameters:
    ----------------------------
    traj          dictionary 
     A dicionary contaning all the cluster objects that are part of a 
     trajectory
     
    Returns
    ----------------------------
    True or False: A boolean value that signifies that verification has been 
    successful or it failed in which case the main code recomputes the 
    trajectories
    """
    #get all the keys which are ids from the dictionary
    listofkeys = list(traj.keys())
    #total number of objects present in the dictionary
    np = len(listofkeys)
    #looping over each object
    for ii in range(0, np):
        #getting the object id
        key = listofkeys[ii]
        #extraction the object
        obj1 = traj.get(key)
        #getting the value of parent
        parent = obj1.parent
        #getting the value of child
        child = obj1.child
        
        #processing parent information
        if parent != 0:
            #if parent present then get the object
            parent_ob = traj.get(parent)
            #check the children of the parent
            children = parent_ob.child
            #looping over all the children present
            if key not in children:
                print('error in verification...')
                print('affected object keys are :', key, parent)
                break
        #processing child information 
        if child != 0:
            #if child present then get the object
            for ii in range(0, len(child)):
                #getting the child id from the list and getting the object
                child_ob = traj.get(child[ii])
                #getting the value of the childs parent
                child_p = child_ob.parent
                if child_p not in key:
                    print('error in verification...')
                    print('affected object keys are:', key, child[ii])
                    break
    return None