# -*- coding: utf-8 -*-
"""
Created on Sat May 18 12:05:18 2019

@author: Arpan Khandelwal
email: napraarpan@gmail.com
"""
#Built-ins
import os
#standard
import numpy as np

#special
try:
    import cv2
except ImportError:
    print("Cannot find opencv package. Try pip intall opencv-contrib-python")
    

def cluster_detection(video_file, path = None, output_name = None, output_type = None):
    """
    Takes a video as input, applies binary threshold to it to convert each 
    frame to binary then runs it through a contour detection algorithm and 
    then for each frame loops through all the contours and draws the minimum 
    enclosing circle. The center co-ordinates of this circle and it's radius 
    are stored and returned for further processing.
    
    It then outputs every frame into the path provided by the user
    
    
    Parameters
    -----------------------
    video_file:       mp4,avi,mpg
     input video along with its path passed in as argument
    path:             string
     Path where the user wants to save the video. By default it take the path 
     from where the raw video file was loaded
    output_name:      String
     Name of the Grayscale converted video. By default it appends to the 
     name of the original file '_clu'
    output_type:      String
     Format of output defined by user. By default it uses the format of the 
     input video
    
    Return
    -----------------------
    """
    #splitting the video file into drive and path + file
    drive, path_file = os.path.splitdrive(video_file)
    #splitting the path + file 
    path_of_file, file = os.path.split(path_file)
    # splitting the file to get the name and the extension
    file = file.split('.')
    
    #checking for the path of the file
    if path is None:
        path = os.path.join(drive,path_of_file)
    #checking for the name of the output file
    if output_name is None:
        output_name = file[0]+'_clu'
    #checking for the putput format of the video
    if output_type is None:
        output_type = '.'+file[1]
    try:
        if os.path.isfile(video_file):
            cap = cv2.VideoCapture(video_file)
    except IOError:
        print("Path or file name incorrect or file does not exist")
        
    #to read the first frame. Returns an error if the videofile has not been loaded correctly
    ret, frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    #get frame width and height of the original video
    #the result video has the same number of pixels
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    #dictionary containing the meta data of the video
    meta_data = {'fps' : fps, 'frame_height' : frame_height, 'frame_width' : frame_width}
    #describing the output file
    pfe = os.path.join(path, output_name + output_type)
    out = cv2.VideoWriter(pfe,fourcc, 25 ,(frame_width,frame_height),0) 
    frame_counter = 1
    while(cap.isOpened()):
        print(frame_counter)
        frame_counter+=1
        ret, img = cap.read()
        img = img[10:640]
        print(img.shape)
        
        #to break out of the loop after exhausting all frames
        if not ret:
            break
        
        print(type(img))
        print(img.shape)
        #Applying the binary threshold method
        ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                127, 255, cv2.THRESH_BINARY)
        
        contours, hierarchy = cv2.findContours(threshed_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            #get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            # draw a green rectangle to visualize the bounding rect
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
            # get the min area rect
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            # convert all coordinates floating point values to int
            box = np.int0(box)
            #draw a red 'nghien' rectangle
            cv2.drawContours(img, [box], 0, (0, 0, 255))
 
            #finally, get the min enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(c)
            #convert all values to int
            center = (int(x), int(y))
            radius = int(radius)
            #and draw the circle in blue
            img = cv2.circle(img, center, radius, (255, 0, 0), 2)
            print(center)
        print(len(contours))
        cv2.drawContours(img, contours, -1, (255, 255, 0), 1)

        out.write(img)
        cv2.imshow("contours", img)
        cv2.imwrite('E:/NERD/Python/data3/frame'+str(frame_counter)+'.jpg',img)
    
    #realeasing outputfile and closing any open windows
    cap.release()
    cv2.destroyAllWindows()
    
    return pfe, meta_data
    
    