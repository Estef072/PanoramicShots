import cv2 as cv
import numpy as np

def homografia(img1, img2, img3):

    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    img3_gray = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    keypoints1, descriptor1 = sift.detectAndCompute(img1_gray, None)
    keypoints2, descriptor2 = sift.detectAndCompute(img2_gray, None)
    keypoints3, descriptor3 = sift.detectAndCompute(img3_gray, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches1_2 = flann.knnMatch(descriptor1, descriptor2, k=2)
    matches2_3 = flann.knnMatch(descriptor2, descriptor3, k=2)

    good_matches1_2 = []
    good_matches2_3 = []

    for m, n in matches1_2:
        if m.distance < 0.7 * n.distance:
            good_matches1_2.append(m)

    for m, n in matches2_3:
        if m.distance < 0.7 * n.distance:
            good_matches2_3.append(m)

    MIN_MATCH_COUNT = 10

    if len(good_matches1_2) > MIN_MATCH_COUNT and len(good_matches2_3) > MIN_MATCH_COUNT:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches1_2]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches1_2]).reshape(-1, 1, 2)

        M1, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        src_pts = np.float32([keypoints2[m.queryIdx].pt for m in good_matches2_3]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints3[m.trainIdx].pt for m in good_matches2_3]).reshape(-1, 1, 2)

        M2, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        rows1, cols1 = img1.shape[:2]
        rows3, cols3 = img3.shape[:2]

        dst1 = cv.warpPerspective(img1, M1, (cols1 + cols3, rows1))
        dst1[:, cols1:] = img3

        rows, cols = img2.shape[:2]
        dst2 = cv.warpPerspective(dst1, M2, (cols + cols3, rows1 + rows3))

        img_panorama = dst2

        return img_panorama
    else:
        print("No se encontraron suficientes coincidencias para realizar la panorámica.")
        return None

if __name__ == '__main__':
    # Leer las tres imágenes
    img1 = cv.imread('pictures/1.png')
    img2 = cv.imread('pictures/2.png')
    img3 = cv.imread('pictures/3.png')

    # Realizar la panorámica
    panorama = homografia(img1, img2, img3)

    # Mostrar el resultado
    if panorama is not None:
        cv.imshow('Panorama Resultante', panorama)
        cv.waitKey(0)
        cv.destroyAllWindows()
