# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 08:40:54 2019

@author: napra
"""

from pynput import Listener

def on_move(x,y):
    pass

def on_click(x,y,button,passed):
    pass
    
def on_scroll(x,y,dx,dy):
    pass
    
with Listener(on_move=on_move, on_click = on_click, on_scroll = on_scroll) as listener:
    listener.join()
    