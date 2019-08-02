from distutils.core import setup
from Cython.Build import cythonize
import numpy

files = ["_criterion.pyx",
         "_utils.pyx",
         "_splitter.pyx",
         "_tree.pyx"]

setup(
    ext_modules=cythonize(["rainforest/" + s for s in files], gdb_debug=True),
    include_dirs=[numpy.get_include()], requires=['numpy', 'pandas']
)