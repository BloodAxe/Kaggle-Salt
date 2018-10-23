from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
import numpy as np


def zero_masks_inplace(predictions, zero_mask_vector):
    predictions[zero_mask_vector] = 0
    return predictions


def morphology_postprocess(image: np.ndarray, mask: np.ndarray, min_size=0):
    # if image.std() == 0:
    ## For plain images predict zero mask
    # return np.zeros_like(mask)

    mask = binary_fill_holes(mask)
    if min_size > 0:
        mask = remove_small_objects(mask, min_size)
    return mask.astype(np.uint8)


def crf_postprocess(image, mask):
    """
    https://github.com/lucasb-eyer/pydensecrf/blob/master/examples/Non%20RGB%20Example.ipynb
    :param image:
    :param mask:
    :return:
    """
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

    H, W = image.shape[:2]

    probs = np.tile(mask[np.newaxis, :, :], (2, 1, 1))
    probs[1, :, :] = 1 - probs[0, :, :]

    U = unary_from_softmax(probs)
    pairwise_energy = create_pairwise_bilateral(sdims=(10, 10), schan=(0.01,), img=image, chdim=-1)

    d = dcrf.DenseCRF2D(W, H, 2)
    d.setUnaryEnergy(U)
    d.addPairwiseEnergy(pairwise_energy, compat=1)  # `compat` is the "strength" of this potential.

    # Run inference for 10 iterations
    Q_unary = d.inference(10)

    # The Q is now the approximate posterior, we can get a MAP estimate using argmax.
    map_soln_unary = 1 - np.argmax(Q_unary, axis=0)

    # Unfortunately, the DenseCRF flattens everything, so get it back into picture form.
    map_soln_unary = map_soln_unary.reshape((H, W))
    return map_soln_unary
