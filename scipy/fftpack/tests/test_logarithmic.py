#!/usr/bin/env python
# Created by Dieter Werthmüller, January 2017

from __future__ import division, print_function, absolute_import

__usage__ = """
Build fftpack:
  python setup.py install
Run tests if scipy is installed:
  python -c 'from scipy import fftpack; fftpack.test(<level>)'
Run tests if fftpack is not installed:
  python tests/test_helper.py [<level>]
"""

from numpy.testing import (TestCase, assert_array_almost_equal,
                           run_module_suite, assert_equal, assert_)
from scipy.fftpack import fftlogargs

from numpy import pi


class TestFFTLogargs(TestCase):

    def test_definition(self):
        # Default values in fftlogargs:
        # - dlogr : 0.01
        # - logrc : 0.0
        # - mu : 0.5
        # - q : 0
        # - kr : 1
        # - kropt : 0

        # Test 1, even n, kr = 2
        inppts = [0.96605088, 0.98855309, 1.01157945, 1.03514217]
        outpts = [1.93210176, 1.97710619, 2.02315891, 2.07028433]
        kr = 2.0
        rk = 0.5
        out = fftlogargs(n=4, kr=2)
        assert_array_almost_equal(inppts, out[0])
        assert_array_almost_equal(outpts, out[1])
        assert_equal(kr, out[2])
        assert_equal(rk, out[3])

        # Test 2, even n, kropt = 1
        inppts = [0.96605088, 0.98855309, 1.01157945, 1.03514217]
        outpts = [0.97306236, 0.9957279, 1.01892138, 1.04265511]
        kr = 1.0072578812188107
        rk = 0.99279441605358465
        out = fftlogargs(n=4, kropt=1)
        assert_array_almost_equal(inppts, out[0])
        assert_array_almost_equal(outpts, out[1])
        assert_equal(kr, out[2])
        assert_equal(rk, out[3])

        # Test 3, odd n, kr = pi, mu = -1
        inppts = [0.95499259, 0.97723722, 1., 1.02329299, 1.04712855]
        outpts = [3.00019769, 3.07008127, 3.14159265, 3.21476975, 3.28965135]
        kr = 3.141592653589793
        rk = 0.31830988618379069
        out = fftlogargs(5, mu=-0.5, kr=pi)
        assert_array_almost_equal(inppts, out[0])
        assert_array_almost_equal(outpts, out[1])
        assert_equal(kr, out[2])
        assert_equal(rk, out[3])

        # Test 4, odd n, logrc = 1, kropt=1
        inppts = [9.54992586, 9.77237221, 10., 10.23292992, 10.47128548]
        outpts = [0.09619238, 0.09843299, 0.10072579, 0.10307199, 0.10547285]
        kr = 1.0072578812188107
        rk = 99.279441605358485
        out = fftlogargs(n=5, logrc=1, kropt=1)
        assert_array_almost_equal(inppts, out[0])
        assert_array_almost_equal(outpts, out[1])
        assert_equal(kr, out[2])
        assert_equal(rk, out[3])

if __name__ == "__main__":
    run_module_suite()
