import cv2
import numpy as np
import os

img = cv2.imread('stop_to_ped_attack.png')
print(img.shape)
masked_img = cv2.imread('stop_to_ped_mask.png')

masked_img = cv2.resize(masked_img,(244,244))
masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
masked_img = masked_img.astype('uint8')




resultimage = cv2.bitwise_and(img, img, mask = masked_img)
resultimage[masked_img == 0] =255

cv2.imwrite('/usr/users/attaquesetdefensessurlia/mangineli/GRAPHITE/example_outputs/stop_to_ped_attack_extracted.png',resultimage)

