from unittest import TestCase

from ncpyramid.pyramid import get_geo_spatial_rect
import numpy as np


class GeoRectTest(TestCase):
    def test_it(self):
        self.assertEqual(get_geo_spatial_rect(np.array([1, 2, 3, 4, 5, 6]),
                                              np.array([1, 2, 3])),
                         (0.5, 0.5, 6.5, 3.5))

        self.assertEqual(get_geo_spatial_rect(np.array([1, 2, 3, 4, 5, 6]),
                                              np.array([3, 2, 1])),
                         (0.5, 3.5, 6.5, 0.5))

        self.assertEqual(get_geo_spatial_rect(np.array([-3, -2, -1, 0, 1, 2]),
                                              np.array([3, 2, 1])),
                         (-3.5, 3.5, 2.5, 0.5))

        self.assertEqual(get_geo_spatial_rect(np.array([177, 178, 179, -180, -179, -178]),
                                              np.array([3, 2, 1])),
                         (176.5, 3.5, 182.5, 0.5))

        self.assertEqual(get_geo_spatial_rect(np.array([-150., -90., -30., 30., 90., 150.]),
                                              np.array([-60., 0., 60.])),
                         (-180.0, -90.0, 180.0, 90.0))
        self.assertEqual(get_geo_spatial_rect(np.array([-150., -90., -30., 30., 90., 150.]),
                                              np.array([60., 0., -60.])),
                         (-180.0, 90.0, 180.0, -90.0))

        eps = 1e-4
        eps025 = 0.25 * eps
        self.assertEqual(get_geo_spatial_rect(
            np.array([-150. - eps025, -90. + eps025, -30. + eps025, 30. - eps025, 90. - eps025, 150. + eps025]),
            np.array([-60. - eps025, 0. - eps025, 60. - eps025]), eps=eps),
                         (-180.0, -90.0, 180.0, 90.0))
