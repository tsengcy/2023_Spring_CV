import numpy as np
import cv2.ximgproc as xip
import cv2


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency
    
    window = 9  #this is the kernel size for calculate the census cost
    padding_size = window//2
    maxvalue = window * window
    imgGrayL = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY)
    imgGrayR = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY)

    imgGrayLPad = cv2.copyMakeBorder(imgGrayL, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_REPLICATE)
    imgGrayRPad = cv2.copyMakeBorder(imgGrayR, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_REPLICATE)

    imgLCensus = np.zeros((h, w, maxvalue)).astype(np.bool) #window*window-1
    imgRCensus = np.zeros((h, w, maxvalue)).astype(np.bool)

    for i in range(window*window):
        # >= is 0
        # < is 1
        # skip the center pixel
        # if i >= (window+1) * (window//2):
        #     width = (i+1) % window
        #     height = (i+1) // window
        # else:
        #     width = i % window
        #     height = i // window
        height = i % window
        width = i // window
        imgLCensus[:, :, i] = imgGrayLPad[height:h+height, width:w+width] < imgGrayL[0:h, 0:w]
        imgRCensus[:, :, i] = imgGrayRPad[height:h+height, width:w+width] < imgGrayR[0:h, 0:w]
    # print(imgLCensus.shape)
    
    costL = np.ones((h, w, max_disp+1)).astype(np.float32) * maxvalue
    costR = np.ones((h, w, max_disp+1)).astype(np.float32) * maxvalue
    for i in range(max_disp+1):
        #平移的問題
        costL[:, i:, i] = np.logical_xor(imgLCensus[:, i:], imgRCensus[:, :(-i) or None]).sum(-1)
        costR[:, :(-i) or None, i] = np.logical_xor(imgLCensus[:, i:], imgRCensus[:, :(-i) or None]).sum(-1)

    # print(costL.shape)
    # print("h", h, "\t w", w)
    # print(imgGrayL.shape)

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    for i in range(max_disp + 1):
        costL[:, :, i] = xip.jointBilateralFilter(imgGrayL, costL[:, :, i], 16, 0.7, 16)
        costR[:, :, i] = xip.jointBilateralFilter(imgGrayR, costR[:, :, i], 16, 0.7, 16)


    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all

    # winner = np.zeros((h,w))
    winnerIndexL = np.zeros((h,w))
    # winner[:,:] = np.amax(costL, axis=2)
    winnerIndexL[:, :] = np.argmin(costL, axis=2)
    winnerIndexR = np.zeros((h,w))
    winnerIndexR[:, :] = np.argmin(costR, axis=2)
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    count = 0
    check = np.zeros((h, w))
    # checking hole = -1
    for i in range(h):
        for j in range(w):
            if (winnerIndexL[i, j] == winnerIndexR[i, int(j-winnerIndexL[i, j])]):
                check[i, j] = 1
                count += 1
    # print(count)
    # print(winnerIndexL[:,0])
    FfromL = winnerIndexL
    FfromR = winnerIndexL
    for i in range(h):
        if(check[i, 0] != 1):
            FfromL[i, 0] = max_disp
        if(check[i, w-1] != 1):
            FfromR[i, w-1] = max_disp
        for j in range(1, w):
            if(check[i, j] != 1):
                FfromL[i, j] = FfromL[i, j-1]
            if(check[i, w-1-j] != 1):
                FfromR[i, w-1-j] = FfromR[i, w-j]


    mylabels = np.minimum(FfromL, FfromR)

    labels = xip.weightedMedianFilter(Il.astype(np.uint8), mylabels.astype(np.float32), 12, 24)



    return labels.astype(np.uint8)