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
        #look up table
        lookUpTable = np.exp(-np.arange(256) * np.arange(256) / (2*self.sigma_r**2 * 255**2))
        
        x, y = np.meshgrid(np.arange(self.wndw_size) - self.pad_w, np.arange(self.wndw_size) - self.pad_w)

        kernel_s = np.exp(-(x * x + y * y) / (2*self.sigma_s**2))

        if guidance.ndim == 3:
            jFlag = True
        else:
            jFlag = False


        # normalize
        
        width = img.shape[1]
        height = img.shape[0]

        imgAfter = np.empty((height, width, 3), float)

        # if not jFlag:
        #     for i in range(self.pad_w, self.pad_w+height):
        #         for j in range(self.pad_w, self.pad_w+width):
        #             weight = lookUpTable[abs(padded_guidance[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1]- padded_guidance[i, j])] * kernel_s
        #             denominator = np.sum(weight)
        #             weight = weight / denominator
        #             imgAfter[i-self.pad_w, j-self.pad_w, 0] = np.sum(weight*padded_img[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1, 0])
        #             imgAfter[i-self.pad_w, j-self.pad_w, 1] = np.sum(weight*padded_img[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1, 1])
        #             imgAfter[i-self.pad_w, j-self.pad_w, 2] = np.sum(weight*padded_img[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1, 2])
        # else:
        #     for i in range(self.pad_w, self.pad_w+height):
        #         for j in range(self.pad_w, self.pad_w+width):
        #             w = lookUpTable[abs(padded_guidance[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1]- padded_guidance[i, j])]
                    
        #             weight = w[:,:,0] * w[:,:,1] * w[:,:,2] * kernel_s
        #             denominator = np.sum(weight) 
        #             weight = weight / denominator
        #             imgAfter[i-self.pad_w, j-self.pad_w, 0] = np.sum(weight*padded_img[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1, 0]) 
        #             imgAfter[i-self.pad_w, j-self.pad_w, 1] = np.sum(weight*padded_img[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1, 1]) 
        #             imgAfter[i-self.pad_w, j-self.pad_w, 2] = np.sum(weight*padded_img[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1, 2]) 


        if not jFlag:
            for i in range(self.pad_w, self.pad_w+height):
                for j in range(self.pad_w, self.pad_w+width):
                    weight = lookUpTable[abs(padded_guidance[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1]- padded_guidance[i, j])] * kernel_s
                    denominator = np.sum(weight)
                    weight = weight / denominator
                    imgAfter[i-self.pad_w, j-self.pad_w, 0] = np.sum(weight*padded_img[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1, 0])
                    imgAfter[i-self.pad_w, j-self.pad_w, 1] = np.sum(weight*padded_img[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1, 1])
                    imgAfter[i-self.pad_w, j-self.pad_w, 2] = np.sum(weight*padded_img[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1, 2])
        else:
            for i in range(self.pad_w, self.pad_w+height):
                for j in range(self.pad_w, self.pad_w+width):
                    w = lookUpTable[abs(padded_guidance[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1]- padded_guidance[i, j])]
                    
                    weight = w[:,:,0] * w[:,:,1] * w[:,:,2] * kernel_s
                    denominator = np.sum(weight) 
                    weight = weight / denominator
                    imgAfter[i-self.pad_w, j-self.pad_w, 0] = np.sum(weight*padded_img[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1, 0]) 
                    imgAfter[i-self.pad_w, j-self.pad_w, 1] = np.sum(weight*padded_img[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1, 1]) 
                    imgAfter[i-self.pad_w, j-self.pad_w, 2] = np.sum(weight*padded_img[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1, 2]) 
                
        output = imgAfter
        
        return np.clip(output, 0, 255).astype(np.uint8)