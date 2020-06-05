from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import numpy as np
from security import Encryption


class TestSecurity(unittest.TestCase):

    # Test permutation functions on 1-dimensional array
    def test_1d_permutation(self):
        numpy_array = np.array([1, 2, 7, -1, 0], np.int32)
        permuted_array = Encryption.p(numpy_array)

        # Check original array is unchanged
        self.assertEqual([1, 2, 7, -1, 0], numpy_array.tolist())

        unpermuted_array = Encryption.un_p(permuted_array)
        self.assertEqual(numpy_array.tolist(), unpermuted_array.tolist())

    # Test permutation functions on 2-dimensional array
    def test_2d_permutation(self):
        numpy_array = np.array([[1, 7, 3], [4, -5, 6]], np.int32)
        permuted_array = Encryption.p(numpy_array)

        # Check original array is unchanged
        self.assertEqual( [[1, 7, 3], [4, -5, 6]], numpy_array.tolist() )

        unpermuted_array = Encryption.un_p( permuted_array )
        self.assertEqual( numpy_array.tolist(), unpermuted_array.tolist() )


