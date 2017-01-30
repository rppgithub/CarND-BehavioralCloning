import cv2

def crop_image(img):
    crop_img = img[60:140,0:320]
    return crop_img

def image_preprocessing(img,img_cols,img_rows):
    """keep only S channel in HSV color space, and resize to 16X32"""
    #cropped_img = crop_image(img)
    resized = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:,:,1],(img_cols,img_rows))

    return resized
