#contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#for c in contours:
#    rect = cv2.boundingRect(c)
#    if rect[2] < 100 or rect[3] < 100: continue
#    print cv2.contourArea(c)
#    x,y,w,h = rect
#    cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
#    cv2.putText(im,'Moth Detected',(x+w+10,y+h),0,0.3,(0,255,0))
#cv2.imshow("Show",im)
#cv2.waitKey()  
#cv2.destroyAllWindows()




import cv2

import numpy as np 
 
# read and scale down image
img = cv2.imread('E:/NERD/Python/DUMPRO/KSTAR_frground/frame437.jpg', cv2.IMREAD_UNCHANGED)
 
# threshold image
blur = cv2.GaussianBlur(img , (11,11),0)
ret, threshed_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# find contours and get the external one
contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
 
frame_center=[]
# with each contour, draw boundingRect in green
# a minAreaRect in red and
# a minEnclosingCircle in blue
for c in contours:
    
    M = cv2.moments(c)
    print('################################################################')
    #print(M)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        center = cx,cy
        print( center)
        area = cv2.contourArea(c)
        print(area)
#    
#    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # draw a green rectangle to visualize the bounding rect
    #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
    # get the min area rect
    rect = cv2.minAreaRect(c)
    print(rect)
    box = cv2.boxPoints(rect)
    # convert all coordinates floating point values to int
    box = np.int0(box)
    # draw a red 'nghien' rectangle
    cv2.drawContours(img, [box], 0, (0, 255, 255))
 
    if M['m00'] == 0:
        # finally, get the min enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(c)
        # convert all values to int
        center = (int(x), int(y))
        radius = int(radius)
        area = 3.14*radius*radius
        print(center)
        print(area)
    # and draw the circle in blue
#    img = cv2.circle(img, center, radius, (255, 0, 0), 2)
    #ellipse = cv2.fitEllipse(c)
    #cv2.ellipse(img,ellipse,(0,255,0),2)
    
    frame_center.append(center)
print(len(contours))
print(type(contours))
#cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
 
cv2.imwrite('E:/NERD/Python/test2.jpg', img)
cv2.destroyAllWindows()
#Use cv2.boundingRect to get the bounding rectangle (in green), 
#cv2.minAreaRect to get the minimum area rectangle (in red), and
# cv2.minEnclosingCircle to get minimum enclosing circle (in blue).
#
#Result:
#
#contour_all.png