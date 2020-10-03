from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
from datetime import datetime
from matplotlib import pyplot as plt
ix, iy = -1, -1
drawing = False  # true if mouse is pressed
  # if True, draw rectangle. Press 'm' to toggle to curve
mode = True

def markup(file):
    rez = []
    rframe = cv.imread(file)


    def draw_circle(event,x,y,flags,param):
      global ix,iy,drawing,mode

      if event == cv.EVENT_LBUTTONDOWN:
          drawing = True
          ix,iy = x,y

      elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                # cv2.rectangle(rframe,(ix,iy),(x,y),(0,255,0),3)
                q=x
                w=y
            #     if q!=x|w!=y:
            #          cv2.rectangle(rframe,(ix,iy),(x,y),(0,0,0),-1)
            # else:
            #     cv2.circle(rframe,(x,y),5,(0,0,255),-1)

      elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv.rectangle(rframe,(ix,iy),(x,y),(0,255,0),2)

        else:
            cv.circle(rframe,(x,y),5,(0,0,255),-1)
        print(x, y, ix, iy)
        rez.append((x,y,ix,iy))

    cv.namedWindow('image')
    cv.setMouseCallback('image',draw_circle)

    while(1):
        rframe = cv.resize(rframe, (1200,800))
        cv.imshow('image',rframe)
        k = cv.waitKey(1) & 0xFF
        if k == ord('m'):
           mode = not mode
        elif k == 27:
           break
    #cv.imwrite('img with rect.png',rframe)
    cv.destroyAllWindows()
    return rez






f1 = open('coord2.txt', 'w')

for i in range(0, 20):
    path = 'Photos1/photo'+str(i)+'.jpg'
    o = markup(path)
    for j in range(0,4):
        f1.write(str(o[0][j])+'\t')
    f1.write('\n')
f1.close()