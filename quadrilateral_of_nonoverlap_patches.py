# Created by: Ashkan Mokarian (firstname.lastname@gmail.com)

"""
Script to return quadrilateral of non-overlapping patches with highest
average brightness in image.
"""

import itertools
from typing import List, Tuple

import numpy as np
import cv2 as cv

PATCH_SIZE = 5


def _is_nonoverlap_patches(c: List[Tuple[int, int]],
                           patch_size=PATCH_SIZE) -> bool:
    """Return True if none of the patches around the list of points overlap"""
    if len(c) <= 1:
        return True
    pairs = itertools.combinations(c, 2)
    for c1, c2 in pairs:
        if abs(c1[0] - c2[0]) < patch_size and abs(c1[1] - c2[1]) < patch_size:
            return False
    return True

def _order_pts(pts):
    """Returns either clock-wise or counter-clock-wise order of 4 input points.
    """
    pts = np.array(pts)
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost
    return np.array([tl, tr, br, bl], dtype="float32")

def get_4best_nonoverlap_patches(im: np.ndarray, 
                                 patch_size: int = PATCH_SIZE,
) -> Tuple[List[Tuple[int, int]], float]:
    """Returns 4 best points and the area of the coresponding quarilateral.
    
    Returns a list of 4 pixel coordinates of input image array such that
    the corresponding fixed size patches around them do not overlap and the sum
    of all patch pixels are the highest. In order to find such 4 best pixels, a
    heuristic is used where at each iteration, only the top k candidates are
    considered and there are no feasible solutions, k is incremented by one
    starting from k=4.
    
    Args:
        im: 2D numpy array representing a greyscale image (e.g. the result of
          cv.imread(file_name, cv.IMREAD_GRAYSCALE)).
        patch_size: [Optional] an odd integer that is the size of the patch,
          that is both width and hight of the patch has the same size. Only odd
          values are expected since the definition of center pixel for patches
          with even sizes is not clear. Raises ValueError if this is even.

    Returns:
        A tuple of size 2.
        
        - First element of the tuple is a list of 4 (x, y) coordinate tuples, or
        None if such 4 points could not be found e.g. if image dimensions are
        too small to fit patch_size patches. The order of coordinates are
        similar to opencv image indexing convention (e.g. x is the column index
        of thearray). For example: [(10, 10), (20, 20), (10, 20), (20,10)]
        - Second element of the tuple is the area covered by the resulting
        quadrilateral in pixels.

    Raises:
        ValueError: if patch_size is not odd or it is less than 1.
    """
    if len(im.shape) != 2:
        return None
    half_patch_size = patch_size // 2
    M, N = im.shape
    # early exit if image too small to fit 4 patches
    if (M < patch_size or N < patch_size or 
        M // patch_size * N // patch_size < 4):
        return None
    # (solution needs further discussion) raises exception for even patchsizes
    # because there is no center of patch
    if patch_size // 2 == 0 or patch_size < 1:
        raise ValueError(f'patch_size must be an odd integer number larger',
        f'than zero. Given patch_size value is {patch_size}')

    avg_im = cv.blur(im, (patch_size,patch_size), borderType=cv.BORDER_CONSTANT)
    # removing border pixels that patch can't fit
    avg_im = avg_im[half_patch_size:-half_patch_size,
                    half_patch_size:-half_patch_size]

    sorted_idx = np.unravel_index(np.argsort(avg_im, axis=None), avg_im.shape)
    # changing to descending order
    sorted_idx = list(zip(sorted_idx[0][::-1], sorted_idx[1][::-1]))
    sorted_val = [avg_im[i, j] for i, j in sorted_idx]
    # add +patch_size//2 to get back to original coordinates
    sorted_idx = [(i+half_patch_size, j+half_patch_size) for i, j in sorted_idx]
    
    
    # Since the search for top four highest average patches is NP-Hard and is
    # O(n^4) in terms of computational complexity, here a heuristic is used
    # called top_candidates_search
    def top_candidates_serach():
        """Returns the four patches with the highest sum.
        
        It uses a heuristic to only compare among a fixed length of top
        candidates by increasing the length by 1 at each iteration until a
        valid solution is found, i.e first searches among the top 4 candidates,
        if no feasible solution were found, searches in the top 5 candidates,
        and so on.
        """
        for p4 in range(3, len(sorted_idx)):
            combs3 = itertools.combinations(range(len(sorted_idx[:p4])), 3)
            best_sum = -1
            best_sol = None
            c4 = sorted_idx[p4]

            for p1, p2, p3 in combs3:
                c1 = sorted_idx[p1]
                c2 = sorted_idx[p2]
                c3 = sorted_idx[p3]
                sum_of_4_avg_values = sum(
                    [sorted_val[i] for i in [p1, p2, p3, p4]])
                if (sum_of_4_avg_values > best_sum and
                    _is_nonoverlap_patches([c1, c2, c3, c4])):
                    best_sum = sum_of_4_avg_values
                    best_sol = [c1, c2, c3, c4]
            if best_sum > -1:
                return best_sol
        return None
    opt_coords = top_candidates_serach()

    # to get a closed shape from cv.polylines, we need to order the coordinates
    # so we dont get zig-zaggy shapes.
    opt_coords = _order_pts(opt_coords)
    # change the order of coordinates to x \times y to adjust for opencv
    # coordinate defaults to use with cv2.contourArea
    opt_coords =  [[y, x] for x, y in opt_coords]
    opt_coords = np.array(opt_coords, dtype=np.int32)
    return opt_coords, cv.contourArea(opt_coords)

def draw_quadrilateral(fn: str, fn_out='./result.png'):
    """Prints Area of the optimal quadrilateral found. Also adds the
    quadrilateral in red to the original image and writes to Disk."""
    im = cv.imread(fn, cv.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError

    opt_coords, area = get_4best_nonoverlap_patches(im)

    print(f'Area = {area}')
    
    # draw the quadrilateral in red
    opt_coords = opt_coords.reshape((-1, 1, 2))
    im = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
    cv.polylines(im, [opt_coords], True, (0, 0, 255))
    
    cv.imwrite(fn_out, im)
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=('Finds 4 non-overlapping average patches of',
        f'size {PATCH_SIZE} in the input image with the highest sum. Draws the',
        'corresponding quadrilateral on the image and writes it to disk, also',
        'prints the area of it.')
    )
    parser.add_argument('image_file', help='File path of the image.')
    parser.add_argument('-o', help=('Output file path. If not assigned, writes',
        'results to "result.png".'), default='result.png', required=False)
    args = parser.parse_args()

    draw_quadrilateral(args.image_file, args.o)

