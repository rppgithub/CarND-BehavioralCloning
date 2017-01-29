import cv2
def image_preprocessing(img,img_cols,img_rows):
    """preproccesing training data to keep only S channel in HSV color space, and resize to 16X32"""
    resized = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:,:,1],(img_cols,img_rows))
    return resized


