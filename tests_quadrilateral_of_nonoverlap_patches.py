import unittest

import numpy as np

import quadrilateral_of_nonoverlap_patches as qonp


def flatten(x):
    """Helper function for assertEqual of nested lists."""
    return [a for b in x for a in b]

class TestIsNonoverlapPatches(unittest.TestCase):
    def test_is_nonoverlap_patches(self):
        self.assertTrue(
            qonp._is_nonoverlap_patches([(10, 100), (100,10), (50, 50)]))
        self.assertFalse(
            qonp._is_nonoverlap_patches([(10, 100), (50, 50), (8, 101)]))

class TestOrderPtsClockwise(unittest.TestCase):
    def test_clockwise(self):
        case = qonp._order_pts([[5, 5], [10, 1], [15, 10], [13, 13]])
        self.assertSequenceEqual(flatten(case),
            flatten(np.array([[10, 1], [15, 10], [13, 13], [5, 5]])))

    def test_counterclockwise(self):
        case =qonp._order_pts([[5, 5], [10, 1], [15, 3], [13, 13]])
        self.assertSequenceEqual(flatten(case),
            flatten(np.array([[10, 1], [15, 3], [13, 13], [5, 5]])))

class TestGet4BestNonoverlapPatches(unittest.TestCase):

    def test_small_square(self):
        def small_square(patch_size=qonp.PATCH_SIZE):
            """Small test case where quadrilateral is a small square and area is
            (patch_size+1)**2.
            """
            im = np.zeros((2*patch_size+1, 2*patch_size+1), dtype=np.uint8) 
            idx = [[i, j] for i in [patch_size - 3, patch_size + 3]
                   for j in [patch_size - 3, patch_size + 3]]
            for i, j in idx:
                im[i-2:i+3,j-2:j+3] = 255
            return im
        coords, area = qonp.get_4best_nonoverlap_patches(small_square())
        self.assertEqual(area, (qonp.PATCH_SIZE+1)**2)
    
    def test_hill(self):
        """Test for naive greedy where best is chosen and would fail here"""
        def hill_from_above(width = qonp.PATCH_SIZE * 4):
            """Image where highest value in center and descent as further away"""
            x, y = np.meshgrid(np.linspace(-1,1,width), np.linspace(-1,1,width))
            d = np.sqrt(x*x+y*y)
            sigma, mu = 1.0, 0.0
            return np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) ) * 255
        coords, area = qonp.get_4best_nonoverlap_patches(hill_from_above())
        self.assertEqual(area, (qonp.PATCH_SIZE)**2)
    
    def test_invalid_image_size(self):
        self.assertIsNone(qonp.get_4best_nonoverlap_patches(
            np.zeros((qonp.PATCH_SIZE-1, 100))))

    def test_input_not_grayscale(self):
        self.assertIsNone(qonp.get_4best_nonoverlap_patches(
            np.zeros((40, 40, 3))))
        

if __name__ == "__main__":
    unittest.main()