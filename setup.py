from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy as np

sources = ["PyClipper/PyClipper.pyx", "PyClipper/clipper.cpp"]


ext = Extension("PyClipper",
                sources=sources,
                language="c++",
                include_dirs=[np.get_include()],
                define_macros=[('use_int32', 1)]
                )

setup(ext_modules=cythonize([ext]))
