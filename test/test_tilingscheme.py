from unittest import TestCase

from ncpyramid.pyramid import TilingScheme

NEG_Y_AXIS_GLOBAL_RECT = (-180., +90., +180., -90.)
POS_Y_AXIS_GLOBAL_RECT = (-180., -90., +180., +90.)

class SubdivisionTest(TestCase):
    def test_cci_ecv(self):
        # Aerosol CCI - monthly
        self.assertEqual(TilingScheme.create(7200, 3600, 500, 500, NEG_Y_AXIS_GLOBAL_RECT),
                         TilingScheme(4, 2, 1, 450, 450, POS_Y_AXIS_GLOBAL_RECT))
        # Cloud CCI - monthly
        self.assertEqual(TilingScheme.create(720, 360, 500, 500, NEG_Y_AXIS_GLOBAL_RECT),
                         TilingScheme(2, 1, 1, 360, 180, POS_Y_AXIS_GLOBAL_RECT))
        # SST CCI - daily L4
        self.assertEqual(TilingScheme.create(8640, 4320, 500, 500, POS_Y_AXIS_GLOBAL_RECT),
                         TilingScheme(4, 2, 1, 540, 540, POS_Y_AXIS_GLOBAL_RECT))
        # Land Cover CCI
        self.assertEqual(TilingScheme.create(129600, 64800, 500, 500, NEG_Y_AXIS_GLOBAL_RECT),
                         TilingScheme(6, 6, 3, 675, 675, POS_Y_AXIS_GLOBAL_RECT))

    def test_subsets(self):
        self.assertEqual(TilingScheme.create(4000, 3000, 500, 500, (-20., 70., 60., 10.)),
                         TilingScheme(4, 1, 1, 500, 375, (-20., 10., 60., 70.)))
        self.assertEqual(TilingScheme.create(4012, 3009, 500, 500, (-20., 70., 60., 10.)),
                         TilingScheme(2, 3, 5, 669, 301, (-20.0, 9.980059820538386, 60.03988035892323, 70.)))
