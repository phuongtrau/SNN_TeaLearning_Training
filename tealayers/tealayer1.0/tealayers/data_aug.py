import cv2
import numpy as np
from scipy.ndimage import zoom

def data_aug(img,rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.5,
        zoom_range=(0.9, 1.1),
        horizontal_flip=False,
        vertical_flip=False):
    h,w = img.shape

    M_height = np.float32([[1, 0, height_shift_range*h],
                            [0, 1, 0],
                            [0, 0, 1]])
    
    M_width = np.float32([[1, 0, 0,
                        [0, 1,  width_shift_range*w],
                        [0, 0, 1]])

    M_shear_1 = np.float32([[1, shear_range, 0],
             	            [0, 1  , 0],
            	            [0, 0  , 1]])

    M_shear_2 = np.float32([[1, 0, 0],
             	            [shear_range, 1  , 0],
            	            [0, 0  , 1]])
    
    
    angle = np.radians(rotation_range)
    
    M_rot = np.float32([[np.cos(angle), -(np.sin(angle)), 0],
            	        [np.sin(angle), np.cos(angle), 0],
            	        [0, 0, 1]])
    img_shift_height = cv2.warpPerspective(img, M_height, (int(h),int(w))))
    img_shift_width = cv2.warpPerspective(img, M_width, (int(h),int(w))))
    img_shear_1 = cv2.warpPerspective(img, M_shear_1, (int(h),int(w))))
    img_shear_2 = cv2.warpPerspective(img, M_shear_2, (int(h),int(w))))
    img_rot = cv2.warpPerspective(img, M_rot, (int(h),int(w))))

    img_zoom_out = clipped_zoom(img,zoom_range[1])
    img_zoom_in = clipped_zoom(img,zoom_range[0])
    
    return np.concatenate([img_shift_height[np.newaxis,:,:],img_shift_width[np.newaxis,:,:],\
        img_shear_1[np.newaxis,:,:],img_shear_2[np.newaxis,:,:],img_rot[np.newaxis,:,:],\
            img_zoom_in[np.newaxis,:,:],img_zoom_out[np.newaxis,:,:]])

def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out

