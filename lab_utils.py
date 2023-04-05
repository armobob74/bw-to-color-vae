from skimage import color
from PIL import Image
import numpy as np

#def get_lab(img):
#    img_array = np.array(img)
#    img_lab = color.rgb2lab(img_array)
#    L = img_lab[:,:,0]
#    a = img_lab[:,:,1]
#    b = img_lab[:,:,2]
#    return np.array([L, a, b])
#
#def get_ab(img):
#    img_array = np.array(img)
#    img_lab = color.rgb2lab(img_array)
#    a = img_lab[:,:,1]
#    b = img_lab[:,:,2]
#    return np.array([a, b])
#
#def get_L(img):
#    img_array = np.array(img)
#    img_lab = color.rgb2lab(img_array)
#    L = img_lab[:,:,0]
#    return np.array(L)
#
#
#
#def img_from_lab(L,a,b):
#    if type(L) == int:
#        z =np.zeros((a.shape[0],a.shape[1],3))
#    else:
#        z =np.zeros((L.shape[0],L.shape[1],3))
#    z[:,:,0] = L
#    z[:,:,1] = a
#    z[:,:,2] = b
#    img_rgb = color.lab2rgb(z, channel_axis=0)
#    return Image.fromarray((img_rgb * 255).astype(np.uint8))
#
#def predisplay_L(L):
#    return img_from_lab(L,0,0)
#
#def predisplay_ab(ab):
#    a, b = ab
#    return img_from_lab(0,a,b)
#

def scale_lab(lab):
    """
    input: CIELAB colorspace tensor of shape [2,H,W]
        range of a and b is -128 to 127
        range of L is 0 to 100

    output: scaled CIELAB colorspace Tensor of shape [3,H,W]
        range of each channel is 0 to 1
    """
    lab[0] = lab[0] / 100
    lab[1] = (lab[1] + 128) / 255
    lab[2] = (lab[2] + 128) / 255
    return lab

def descale_lab(lab):
    """
    input: scaled CIELAB colorspace Tensor of shape [3,H,W]
        range of each channel is 0 to 1

    output: CIELAB colorspace tensor of shape [3,H,W]
        range of a and b is -128 to 127
        range of L is 0 to 100

    Must not edit image in place, otherwise each display of image would change it!
    """
    ret = np.zeros_like(lab)
    ret[0] = lab[0] * 100
    ret[1] = (lab[1] * 255) - 128
    ret[2] = (lab[2] * 255) - 128
    return ret
