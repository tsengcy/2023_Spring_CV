import numpy as np
import cv2
from matplotlib import pyplot as plt

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
        imgCopy = image.copy()
        for _ in range(self.num_octaves):
            # two octave
            gaussian_images.append(imgCopy)
            for j in range(1, self.num_guassian_images_per_octave):
                #five image
                blur = cv2.GaussianBlur(imgCopy, (0,0), self.sigma**j, self.sigma**j)
                gaussian_images.append(blur)
            newWidth = int(imgCopy.shape[1]/2)
            newHeight = int(imgCopy.shape[0]/2)
            imgCopy = cv2.resize(gaussian_images[-1], (newWidth, newHeight), interpolation = cv2.INTER_NEAREST)
        
        
        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for i in range(self.num_octaves):
            for j in range(self.num_DoG_images_per_octave):
                dog_images.append(cv2.subtract(gaussian_images[self.num_guassian_images_per_octave*i + j + 1], gaussian_images[self.num_guassian_images_per_octave*i + j]))

        # for i in range(len(dog_images)):
        #     plt.figure(i)
        #     plt.imshow(dog_images[i], cmap="gray")
        # plt.show()
        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        memo = np.empty((0,2), int)
        for i in range(self.num_octaves):
            # two octave
            # there will drop out 
            oc = np.stack(dog_images[i*self.num_DoG_images_per_octave:(i+1)*self.num_DoG_images_per_octave], axis=-1)
            for j in range(1, self.num_DoG_images_per_octave-1):
                for k in range(1, oc.shape[0]-1):
                    for l in range(1, oc.shape[1]-1):
                        if( abs(oc[k, l, j]) < self.threshold):
                            continue
                        window = oc[k-1:k+2, l-1:l+2, j-1:j+2]
                        if(np.amax(window) == oc[k, l, j] or np.amin(window) == oc[k, l, j]):
                            memo = np.append(memo, np.array([[int((k)*(2**i)), int((l)*(2**i)) ]]), axis=0)
                            
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(memo, axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints
