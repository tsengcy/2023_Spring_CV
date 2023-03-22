import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter

import csv
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    num = 1
    with open(args.setting_path) as f:
        raw = csv.reader(f)
        setting = list(raw)
    sigma_s = int(setting[-1][1])
    sigma_r = float(setting[-1][3])
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    imgGuide = []
    imgGuide.append(img_gray)

    width = img_rgb.shape[1]
    height = img_rgb.shape[0]

    for i in range(1, len(setting)-1):
        weight = np.array(setting[i], float)
        img = np.tensordot(weight, img_rgb, axes=([0], [2]))
        # print(img)
        img = img.astype('uint8')
        imgGuide.append(img)

        
    
    valMax = 0
    indexMax = 0
    valMin = 0
    indexMin = 0
    errorlist = []
    imgOutbf = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
    plt.figure()
    plt.imshow(imgOutbf)
    path = "./output/img" + str(num) + "_bf" + ".png"
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

    for i in range(len(imgGuide)):
        print("#", i)
        plt.figure()
        plt.imshow(imgGuide[i], cmap='gray')
        path = "./output/img" + str(num) + "_filter_no" + str(i) + ".png"
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        imgOutjbf = JBF.joint_bilateral_filter(img_rgb, imgGuide[i]).astype(np.uint8)
        error = np.sum(np.abs(imgOutbf.astype('int32')-imgOutjbf.astype('int32')))
        errorlist.append(error)

        plt.figure()
        plt.imshow(imgOutjbf)
        path = "./output/img" + str(num) + "_no" + str(i) + ".png"
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()

        if(i == 0):
            valMax = error
            valMin = error
        else:
            if(error > valMax):
                valMax = error
                indexMax = i
            elif(error < valMin):
                valMin = error
                indexMin = i
    print("minimum: # {} = {}".format(indexMin, valMin))
    print("maximum: # {} = {}".format(indexMax, valMax))
    print("errorlist")
    print(errorlist)


if __name__ == '__main__':
    main()

