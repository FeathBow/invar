import sys
import unittest

if sys.platform != "darwin":
    raise unittest.SkipTest("Native MLX tests require macOS")
