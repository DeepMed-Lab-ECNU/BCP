import numpy as np
from scipy.ndimage import gaussian_filter
import skimage.measure as skmeasure
import scipy.ndimage as ndi
import torch


class BBoxException(Exception):
    pass


def get_non_empty_min_max_idx_along_axis(mask, axis):
    """
    Get non zero min and max index along given axis.
    :param mask:
    :param axis:
    :return:
    """
    if isinstance(mask, torch.Tensor):
        # pytorch is the axis you want to get
        nonzero_idx = (mask != 0).nonzero()
        if len(nonzero_idx) == 0:
            min = max = 0
        else:
            max = nonzero_idx[:, axis].max()
            min = nonzero_idx[:, axis].min()
    elif isinstance(mask, np.ndarray):
        nonzero_idx = (mask != 0).nonzero()
        if len(nonzero_idx[axis]) == 0:
            min = max = 0
        else:
            max = nonzero_idx[axis].max()
            min = nonzero_idx[axis].min()
    else:
        raise BBoxException("Wrong type")
    max += 1
    return min, max


def get_bbox_3d(mask):
    """ Input : [D, H, W] , output : ((min_x, max_x), (min_y, max_y), (min_z, max_z))
    Return non zero value's min and max index for a mask
    If no value exists, an array of all zero returns
    :param mask:  numpy of [D, H, W]
    :return:
    """
    assert len(mask.shape) == 3
    min_z, max_z = get_non_empty_min_max_idx_along_axis(mask, 0)
    min_y, max_y = get_non_empty_min_max_idx_along_axis(mask, 1)
    min_x, max_x = get_non_empty_min_max_idx_along_axis(mask, 2)

    return np.array(((min_x, max_x + 1),
                     (min_y, max_y + 1),
                     (min_z, max_z + 1)))


def pad_bbox(bbox, min_bbox, max_img):
    """
    :param bbox:  ndarray ((min_x, max_x), (min_y, max_y), (min_z, max_z))
    :param min_bbox: list (d, h, w)
    :param max_img:  list (d, h,  w), image shape
    :return:
    """
    min_bbox = list(min_bbox)
    change_min_bbox = False
    for i, (min_x, max_img_x) in enumerate(zip(min_bbox, max_img)):
        if min_x > max_img_x:
            min_bbox[i] = max_img[i]
            change_min_bbox = True

    if change_min_bbox:
        print('min box {} is larger than max image size {}'.format(min_bbox, max_img))

    # z first
    bbox = np.array(bbox)[::-1, :]
    result_bbox = []
    for (min_x, max_x), min_size, max_size in zip(bbox, min_bbox, max_img):
        width = max_x - min_x
        if width < min_size:
            padding = min_size - width
            padding_left = padding // 2
            padding_right = padding - padding_left

            # find a best place to pad img
            while True:
                if (min_x - padding_left) < 0 and (max_x + padding_right) > max_size:
                    # pad to img size
                    padding_left = min_x
                    padding_right = max_size - max_x
                    break
                elif (min_x - padding_left) < 0:
                    # right shift pad
                    padding_left -= 1
                    padding_right += 1
                elif (max_x + padding_right) > max_size:
                    # left shift pad
                    padding_left += 1
                    padding_right -= 1
                else:
                    # no operation to pad
                    break
            min_x -= padding_left
            max_x += padding_right
        result_bbox.append((min_x, max_x))
    # x first
    return np.array(result_bbox)[::-1, :]


def expand_bbox(img, bbox, expand_size, min_crop_size):
    img_z, img_y, img_x = img.shape

    # expand [[154 371  15] [439 499  68]]
    bbox[:, 0] -= expand_size[::-1]  # min (x, y, z)
    bbox[:, 1] += expand_size[::-1]  # max (x, y, z)
    # prevent out of range
    bbox[0, :] = np.clip(bbox[0, :], 0, img_x)
    bbox[1, :] = np.clip(bbox[1, :], 0, img_y)
    bbox[2, :] = np.clip(bbox[2, :], 0, img_z)

    # expand, then pad
    bbox = pad_bbox(bbox, min_crop_size, img.shape)
    return bbox



def crop_img(img, bbox, min_crop_size):
    """ Crop image with expanded bbox.
    :param img:  ndarray (D, H, W)
    :param bbox: ndarray ((min_x, max_x), (min_y, max_y), (min_z, max_z))
    :param min_crop_size: list (d, h ,w)
    :return:
    """

    # extract coords
    (min_x, max_x), (min_y, max_y), (min_z, max_z) = bbox

    # crop
    cropped_img = img[min_z:max_z, min_y:max_y, min_x:max_x]

    padding = []
    for i, (cropped_width, min_width) in enumerate(zip(cropped_img.shape, min_crop_size)):
        if cropped_width < min_width:
           padding.append((0, min_width - cropped_width))
        else:
           padding.append((0, 0))
    padding = np.array(padding).astype(np.int)
    cropped_img = np.pad(cropped_img, padding, mode='constant', constant_values=0)
    return cropped_img


from dipy.align.reslice import reslice
def resample_volume_nib(np_data, affine, spacing_old, spacing_new=(1., 1., 1.), mask=False):
    """Resample 3D image(trilinear) and mask(nearest) to (1., 1., 1.) spacing.
       It seems works better than the method above, seen from generated image.

    :param np_data: ndarray, channel first
    :param affine: the affine returned from nibabel
    :param spacing_old:  current spacing
    :param spacing_new: target spacing, default is (1., 1., 1.)
    :param mask: if set True, use nearest instead of trilinear interpolation
    :return:
        resampled data : ndarray
        affine         : the modified affine.
    """
    if not mask:
        # trilinear
        resampled_data, affine = reslice(np_data, affine, spacing_old, spacing_new, order=1)
    else:
        # nearest
        resampled_data, affine = reslice(np_data, affine, spacing_old, spacing_new, order=0)
    return resampled_data, affine