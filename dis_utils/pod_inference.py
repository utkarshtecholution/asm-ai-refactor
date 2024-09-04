from inf import inference
import numpy as np 
import cv2 

xmin,ymin,xmax,ymax=  [120,100,359, 480]

image = cv2.imread("../Images/2c2e46ec-3d12-11ef-a279-4fddbacf87f7.jpg")
cropped_image = image[ymin:ymax, xmin:xmax]
# print(cropped_image.shape)
cv2.imshow('Cropped Image', cropped_image)

# cv2.imshow('original full imge mask', inference(image))
# Generate the binary mask (assuming inference function is defined)
binary_mask = inference(cropped_image)


cv2.imshow("Original mask", binary_mask)

# Create an empty mask with the same size as the original image
mask_overlay = np.zeros(image.shape[:2], dtype=np.uint8)



# Place the generated mask in the correct location within the empty mask
# mask_overlay[ymin:ymax, xmin:xmax] = binary_mask[ymin:ymax, xmin:xmax]

mask_overlay[ymin:ymax, xmin:xmax] = binary_mask
cv2.imshow('Mask', mask_overlay)
if cv2.waitKey(0) & 0xff == ord('q'):
    cv2.destroyAllWindows()