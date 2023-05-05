import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """

    # use the method that set h33 = 1
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A

    # A = np.zeros((2*N, 8))
    A = np.zeros((2*N, 9))
    B = np.zeros((2*N, 1))

    # for i in range(N):
    #     A[2*i, 0] = u[i, 0]
    #     A[2*i, 1] = u[i, 1]
    #     A[2*i, 2] = 1.0

    #     A[2*i+1, 3] = u[i, 0]
    #     A[2*i+1, 4] = u[i, 1]
    #     A[2*i+1, 5] = 1.0

    #     A[2*i, 6] = -1 * u[i, 0] * v[i, 0]
    #     A[2*i, 7] = -1 * u[i, 1] * v[i, 0]

    #     A[2*i+1, 6] = -1 * u[i, 0] * v[i, 1]
    #     A[2*i+1, 7] = -1 * u[i, 1] * v[i, 1]

    #     B[2*i, 0] = v[i, 0]
    #     B[2*i+1, 0] = v[i, 1]
    # print("A\n", A)
    # print("B\n{}".format(B))

    for i in range(N):
        A[2*i, 0] = u[i, 0]
        A[2*i, 1] = u[i, 1]
        A[2*i, 2] = 1

        A[2*i+1, 3] = u[i, 0]
        A[2*i+1, 4] = u[i, 1]
        A[2*i+1, 5] = 1
        
        A[2*i, 6] = -1 * u[i, 0] * v[i, 0]
        A[2*i, 7] = -1 * u[i, 1] * v[i, 0]
        A[2*i, 8] = -1 * v[i, 0]

        A[2*i+1, 6] = -1 * u[i, 0] * v[i, 1]
        A[2*i+1, 7] = -1 * u[i, 1] * v[i, 1]
        A[2*i+1, 8] = -1 * v[i, 1]

    # TODO: 2.solve H with A

    # h = np.linalg.pinv(A).dot(B)
    # print(h.shape)
    # h = h.reshape(-1)
    # print("h\n{}".format(h))

    # H = np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]])
    # print("H\n{}".format(H))
    # print("A", A)
    _, _, V = np.linalg.svd(A)
    # print("V\n",V)
    h = V[-1, :] / V[-1,-1]

    # print("h", h)
    H = h.reshape(3, 3)
    # print("H", H)

    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs

    x = np.arange(xmin, xmax)
    y = np.arange(ymin, ymax)

    ux, uy = np.meshgrid(x, y)
    ux = ux.reshape(-1).astype(int)
    uy = uy.reshape(-1).astype(int)

    # print("ux\n{}".format(ux))
    # print("uy\n{}".format(uy))

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate

    desPixel = np.array([ux, uy, np.ones(ux.shape[0])])
    # print("despixel\n{}".format(desPixel))

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)

        newpixel = H_inv.dot(desPixel)
        vx = np.floor(newpixel[0, :]).flatten().astype(int)
        vy = np.floor(newpixel[1, :]).flatten().astype(int)
        # print(vx)
        # print(vy)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)

        mask = np.where((vx >=0) & (vx < w_src) & (vy >=0) & (vy < h_src))

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates

        ux, uy = ux[mask], uy[mask]
        vx, vy = vx[mask], vy[mask]

        # TODO: 6. assign to destination image with proper masking

        dst[uy, ux] = src[vy, vx]

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        
        newpixel = H.dot(desPixel)
        newpixel = newpixel / newpixel[-1]
        vx = np.floor(newpixel[0, :]).reshape(-1).astype(int)
        vy = np.floor(newpixel[1, :]).reshape(-1).astype(int)
        # print(vx)
        # print(vy)


        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)

        mask = np.where((vx >=0) & (vx < w_dst) & (vy >=0) & (vy < h_dst))


        # TODO: 5.filter the valid coordinates using previous obtained mask

        ux, uy = ux[mask], uy[mask]
        vx, vy = vx[mask], vy[mask]


        # TODO: 6. assign to destination image using advanced array indicing

        dst[vy, vx] = src[uy, ux]


    return dst
