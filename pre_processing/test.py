import cv2
import os
img_name = 'mdb285.pgm'
img_path = os.path.join(r'/home/tinku/myfiles/final_project/review_2/NEW (copy)//MICROCALCIFICATIONS ONLY/my_code',img_name)
print(type(img_path))
print(img_path)
cv2.imshow("pre",img_path)
print('end')