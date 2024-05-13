import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_images(image_dir, image_files):
    """Load images from a specified directory."""
    images = [cv2.imread(f'{image_dir}/{file}') for file in image_files]
    return images

def stitch_images(images):
    """Stitch multiple images into a single panorama."""
    # Convert the first image to grayscale
    current_image = images[0]
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

    # Initialize SIFT
    sift = cv2.SIFT_create()
    
    # Iterate over all images to stitch them together
    for i in range(1, len(images)):
        # Detect keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(gray_images[i-1], None)
        kp2, des2 = sift.detectAndCompute(gray_images[i], None)
        
        # Match features
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        
        # Homography calculation
        if len(good) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            # Warp the new image to the panorama
            h, w = images[i].shape[:2]
            w1, h1 = current_image.shape[:2]
            panorama = cv2.warpPerspective(current_image, H, (w + w1, h))
            panorama[0:h, 0:w] = images[i]
            current_image = panorama
        else:
            raise AssertionError("Can't find enough keypoints.")

    return current_image

def main():
    image_dir = 'pictures2'
    image_files = ['10.jpeg',
                   '9.jpeg',
                   '8.jpeg',
                   '7.jpeg',
                   '6.jpeg',
                   '5.jpeg',
                   '3.jpeg',
                   '2.jpeg',
                   '1.jpeg',
                   
                   ]  
    images = load_images(image_dir, image_files)
    panorama = stitch_images(images)
    
    # Save and show the output
    cv2.imwrite('output.jpg', panorama)
    plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()