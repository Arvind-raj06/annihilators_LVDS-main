import numpy as np
import cv2
import imutils
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
def noplateno(img):
    image=img
    image = imutils.resize(image, width=500)
    cv2.imshow("Original Image", image)
    cv2.waitKey(500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 170, 200)
    cnts, new  = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img1 = image.copy()
    cv2.drawContours(img1, cnts, -1, (0,255,0), 3)
    cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
    NumberPlateCnt = None
    img2 = image.copy()
    cv2.drawContours(img2, cnts, -1, (0,255,0), 3)
    count = 0
    idx =7
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            NumberPlateCnt = approx
            x, y, w, h = cv2.boundingRect(c)
            new_img = gray[y:y + h, x:x + w]
            cv2.imwrite('datasets/crop/' + str(idx) + '.png', new_img)
            idx+=1

            break
    cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)
    cv2.imshow("Final Image With Number Plate Detected", image)
    cv2.waitKey(500)

    Cropped_img_loc = 'datasets/crop/7.png'
    cim=cv2.imread(Cropped_img_loc)
    text = pytesseract.image_to_string(np.array(cim), lang='eng')
    ts=""
    for i in text:
        if((i>='a'and i<='z') or (i>='A' and i<='Z')or(i>='0'and i<='9')):
            ts+=i
    return ts
