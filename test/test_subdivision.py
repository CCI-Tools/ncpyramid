from unittest import TestCase

from ncpyramid.pyramid import pow2_2d_subdivision, pow2_1d_subdivisions


class SubdivisionTest(TestCase):
    def test_size_subdivisions(self):
        # Aerosol CCI - monthly
        self.assertEqual(pow2_1d_subdivisions(360),
                         [(360, 360, 1, 1)])
        self.assertEqual(pow2_1d_subdivisions(180),
                         [(180, 180, 1, 1)])
        # Cloud CCI - monthly
        self.assertEqual(pow2_1d_subdivisions(720),
                         [(720, 360, 1, 2)])
        self.assertEqual(pow2_1d_subdivisions(360),
                         [(360, 360, 1, 1)])
        # SST CCI - daily L4
        self.assertEqual(pow2_1d_subdivisions(7200),
                         [(7200, 225, 1, 6),
                          (7200, 450, 1, 5),
                          (7200, 900, 1, 4),
                          (7200, 225, 2, 5),
                          (7200, 450, 2, 4),
                          (7200, 900, 2, 3),
                          (7200, 300, 3, 4),
                          (7200, 600, 3, 3),
                          (7200, 1200, 3, 2),
                          (7200, 225, 4, 4),
                          (7200, 450, 4, 3),
                          (7200, 900, 4, 2),
                          (7200, 360, 5, 3),
                          (7200, 720, 5, 2),
                          (7200, 300, 6, 3),
                          (7200, 600, 6, 2)])
        self.assertEqual(pow2_1d_subdivisions(3600),
                         [(3600, 225, 1, 5),
                          (3600, 450, 1, 4),
                          (3600, 900, 1, 3),
                          (3600, 225, 2, 4),
                          (3600, 450, 2, 3),
                          (3600, 900, 2, 2),
                          (3600, 300, 3, 3),
                          (3600, 600, 3, 2),
                          (3600, 225, 4, 3),
                          (3600, 450, 4, 2),
                          (3600, 360, 5, 2),
                          (3600, 300, 6, 2)])
        # Land Cover CCI
        self.assertEqual(pow2_1d_subdivisions(129600),
                         [(129600, 675, 3, 7),
                          (129600, 405, 5, 7),
                          (129600, 810, 5, 6),
                          (129600, 675, 6, 6)])
        self.assertEqual(pow2_1d_subdivisions(64800),
                         [(64800, 675, 3, 6),
                          (64800, 405, 5, 6),
                          (64800, 810, 5, 5),
                          (64800, 675, 6, 5)])

    def test_rect_subdivision_easy(self):
        # Aerosol CCI - monthly
        self.assertEqual(pow2_2d_subdivision(360, 180), ((360, 180), (360, 180), (1, 1), 1))
        # Cloud CCI - monthly
        self.assertEqual(pow2_2d_subdivision(720, 360), ((720, 360), (360, 360), (2, 1), 1))
        # SST CCI - daily L4
        self.assertEqual(pow2_2d_subdivision(7200, 3600), ((7200, 3600), (225, 225), (2, 1), 5))
        # OD CCI - monthly L3S
        self.assertEqual(pow2_2d_subdivision(8640, 4320), ((8640, 4320), (270, 270), (2, 1), 5))
        self.assertEqual(pow2_2d_subdivision(8640, 4320, tw_opt=1440, th_opt=1440),
                         ((8640, 4320), (1080, 1080), (2, 1), 3))
        # Land Cover CCI
        self.assertEqual(pow2_2d_subdivision(129600, 64800), ((129600, 64800), (675, 675), (6, 3), 6))

    def test_rect_subdivision_hard(self):
        self.assertEqual(pow2_2d_subdivision(4823, 5221),
                         ((4823, 5221), (4823, 5221), (1, 1), 1))
        self.assertEqual(pow2_2d_subdivision(4823, 5221, w_mode=1, h_mode=-1),
                         ((4824, 4180), (603, 1045), (2, 1), 3))
        self.assertEqual(pow2_2d_subdivision(4823, 5221, w_mode=-1, h_mode=1),
                         ((3860, 5222), (965, 373), (2, 7), 2))
        self.assertEqual(pow2_2d_subdivision(4823, 5221, w_mode=-1, h_mode=-1),
                         ((3860, 4180), (965, 1045), (1, 1), 3))
        self.assertEqual(pow2_2d_subdivision(4823, 5221, w_mode=1, h_mode=1),
                         ((4824, 5222), (603, 373), (4, 7), 2))
        self.assertEqual(pow2_2d_subdivision(4823, 5221, w_mode=1, h_mode=1, tw_opt=500, th_opt=500),
                         ((4824, 5222), (603, 373), (4, 7), 2))

        self.assertEqual(pow2_2d_subdivision(934327, 38294, w_mode=1, h_mode=1, tw_opt=500, th_opt=500),
                         ((934400, 38304), (365, 399), (80, 3), 6))
