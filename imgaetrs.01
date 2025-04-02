import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return
    image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
    print(f"Image Dimensions: {image.shape}")
    print(f"Image Size: {image.size} bytes")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayscale, 128, 255, cv2.THRESH_BINARY)

    scaled = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

    denoised = cv2.GaussianBlur(image, (5, 5), 0)

    cv2.imshow("Original Image", image)
    cv2.imshow("Grayscale Image", grayscale)
    cv2.imshow("Binary Image", binary)
    cv2.imshow("Scaled Image", scaled)
    cv2.imshow("Denoised Image", denoised)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = "tree.jpg"
process_image(image_path)
