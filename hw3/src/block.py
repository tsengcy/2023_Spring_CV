import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        imgGray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        imgGray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()

        kp1, des1 = orb.detectAndCompute(imgGray1, None)
        kp1, des2 = orb.detectAndCompute(imgGray2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])

        good = sorted(good, key= lambda x:x.distance)
        v = np.array([kp1[m.queryIdx].pt for m in good])
        u = np.array([kp2[m.queryIdx].pt for m in good])

        # TODO: 2. apply RANSAC to choose best H
        p = 0.99
        e = 0.7
        s = 4
        epoch = int(np.log(1 - p) / np.log(1 - (1 - e) ** s))
        threshold = 4
        inline_num_max = 0
        H_best = np.eye(3)
        for i in tqdm(range(epoch)):            
            idx = np.random.choice(range(len(u)), 4, replace=False)
            us, vs = u[idx], v[idx] 

            H = solve_homography(us, vs)

            U = np.vstack((u.T, np.ones((1, len(u)))))
            V = np.vstack((v.T, np.ones((1, len(v)))))
         
            V_estimate = H.dot(U)
            V_estimate = V_estimate / V_estimate[-1]
            
            distance = np.linalg.norm((V_estimate - V)[:-1,:], ord=1, axis=0)
            inline_num = sum(distance < threshold)
            
            if inline_num > inline_num_max:
                inline_num_max = inline_num
                H_best = H
        
        # TODO: 3. chain the homographies
        last_best_H = last_best_H.dot(H)

        # TODO: 4. apply warping
        out = warping(im2, dst, last_best_H, 0, im2.shape[0], 0, im2.shape[1], direction='b')
        

    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    imgGray1 = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)
    imgGray2 = cv2.cvtColor(imgs[1], cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(imgGray1, None)
    kp2, des2 = orb.detectAndCompute(imgGray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x:x.distance)

    img3 = cv2.drawMatches(imgs[0], kp1, imgs[1], kp2, matches[:10], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow("image", img3)
    cv2.waitKey(10000)

    # output4 = panorama(imgs)
    # cv2.imwrite('output4.png', output4)