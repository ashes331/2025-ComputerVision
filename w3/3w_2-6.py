import cv2 as cv
import sys

img=cv.imread('soccer.jpg') 

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')
    
cv.rectangle(img,(250,50),(330,150),(0,0,255),2) 
# 이미지, 시작점, 끝점좌표, 사각형 BGR 색상, 선 굵기

cv.putText(img, 'face', (250,40), cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
# 이미지, 텍스트 내용, 텍스트 시작점, 폰트, 글자크기, BGR 색상, 글자 선의 두계

cv.imshow('20241491 김성원', img)

cv.waitKey()
cv.destroyAllWindows()
