import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        #normalize the image
        imgOrignal = float(padded_img) / 255
        imgGuide = float(padded_guidance) / 255

        #sptial kernel

        width = img.shape[1]
        height = img.shape[0]

        imgafter = np.empty((height, width), float)

        for i in range(height):
            for j in range(width):
                


        
        return np.clip(output, 0, 255).astype(np.uint8)