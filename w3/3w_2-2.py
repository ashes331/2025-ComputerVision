import cv2 as cv
import sys

img=cv.imread('soccer.jpg') # 영상읽기

print(img[0,0])
print(img[0,0,1])

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')
    
cv.imshow('Image Display', img)

cv.waitKey()
cv.destroyAllWindows()    















