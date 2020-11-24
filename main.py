import no_plate as p
import type as t
import cv2
import colordetect as cd
img=cv2.imread('DVLA-number-plates-2017-67-new-car-847566.jpg')
t.typevehicle(img)
no=p.noplateno(img)
if(len(no)==0):
    print("Number plate not found")
else:
    print("Number is ",no)
color=cd.cdc(img)
if(len(color)==0):
    print("color couldn't be predicted")
else:
    print("color is:",color)
cv2.destroyAllWindows()
