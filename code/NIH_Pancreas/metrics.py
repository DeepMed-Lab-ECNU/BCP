import numpy as np
import torch
import torch.nn.functional as F
#from medpy import metric
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure
from scipy.ndimage import _ni_support
import pdb

def surface_distances(result, reference, voxelspacing=None, connectivity=1):
        """
        The distances between the surface voxel of binary objects in result and their
        nearest partner surface voxel of a binary object in reference.
        """
        result = np.atleast_1d(result.astype(np.bool))
        reference = np.atleast_1d(reference.astype(np.bool))
        if voxelspacing is not None:
            voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
            voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
            if not voxelspacing.flags.contiguous:
                voxelspacing = voxelspacing.copy()

        # binary structure
        footprint = generate_binary_structure(result.ndim, connectivity)

        # test for emptiness
        if 0 == np.count_nonzero(result):
            raise RuntimeError('The first supplied array does not contain any binary object.')
        if 0 == np.count_nonzero(reference):
            raise RuntimeError('The second supplied array does not contain any binary object.')

            # extract only 1-pixel border line of objects
        result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
        reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

        # compute average surface distance
        # Note: scipys distance transform is calculated only inside the borders of the
        #       foreground objects, therefore the input has to be reversed
        dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
        sds = dt[result_border]
        return sds

def hd95(result, reference, voxelspacing=None, connectivity=1):
        """
        95th percentile of the Hausdorff Distance.
        Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
        images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
        commonly used in Biomedical Segmentation challenges.
        Parameters
        ----------
        result : array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
        reference : array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
        voxelspacing : float or sequence of floats, optional
            The voxelspacing in a distance unit i.e. spacing of elements
            along each dimension. If a sequence, must be of length equal to
            the input rank; if a single number, this is used for all axes. If
            not specified, a grid spacing of unity is implied.
        connectivity : int
            The neighbourhood/connectivity considered when determining the surface
            of the binary objects. This value is passed to
            `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
            Note that the connectivity influences the result in the case of the Hausdorff distance.
        Returns
        -------
        hd : float
            The symmetric Hausdorff Distance between the object(s) in ```result``` and the
            object(s) in ```reference```. The distance unit is the same as for the spacing of
            elements along each dimension, which is usually given in mm.
        See also
        --------
        :func:`hd`
        Notes
        -----
        This is a real metric. The binary images can therefore be supplied in any order.
        """
        hd1 = surface_distances(result, reference, voxelspacing, connectivity)
        hd2 = surface_distances(reference, result, voxelspacing, connectivity)
        hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
       
        return hd95

def Jaccord(output, target, numpy=False):
    smooth = 1e-5

    output = output[:,1:,:,:,:]
    target = target[:,1:,:,:,:]

    if not numpy:
      if torch.is_tensor(output):
        output = output.data.cpu().numpy()
      if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_ = output > 0.5
    target_ = target > 0.5

    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target, numpy=True):
    smooth = 1e-8
    N = 1

    output = output[1,1:,:,:,:]
    target = target[1,1:,:,:,:]

    if numpy:
      output = output.reshape(N, -1)
      target = target.reshape(N, -1)
    else:
      output = output.contiguous().view(N, -1).detach().cpu().numpy()
      target = target.contiguous().view(N, -1).detach().cpu().numpy()


    output_ = output > 0.5
    target_ = target > 0.5

    output = output_.astype(int)
    target = target_.astype(int)

    intersection = (output * target).sum(axis=1)

    all_iou = (2. * intersection + smooth) / (output.sum(axis=1) + target.sum(axis=1) + smooth)

    return all_iou.sum()

def HD(output, target):

    output = (output > 0.5)
    target = (target > 0.5)

    # output = output[:,1:,:,:,:].squeeze()
    # target = target[:,1:,:,:,:].squeeze()
 
    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    hd_ = hd95(output, target)

    return np.mean(hd_)

def ASD(output, target):
    output = (output > 0.5)
    target = (target > 0.5)

    # output = output[:,1:,:,:,:].squeeze()
    # target = target[:,1:,:,:,:].squeeze()

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    
    asd_ = surface_distances(output, target)

    return np.mean(asd_)
