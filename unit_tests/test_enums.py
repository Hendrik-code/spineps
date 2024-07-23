# Call 'python -m unittest' on this folder  # noqa: INP001
# coverage run -m unittest
# coverage report
# coverage html
import os
import unittest

from spineps.seg_enums import ErrCode


class Test_Enums(unittest.TestCase):
    def test_errorcode(self):
        print(ErrCode.OK)
        # print(MatchingMetric.DSC.name)

        self.assertEqual(ErrCode.OK, ErrCode.OK)
        self.assertEqual(ErrCode.OK, "OK")
        self.assertEqual(ErrCode.OK.name, "OK")
        self.assertNotEqual(ErrCode.OK, ErrCode.ALL_DONE)
        self.assertNotEqual(ErrCode.UNKNOWN, "OK")
