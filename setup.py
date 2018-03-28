from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

"""
Note on using the setup.py:
setup.py operates in 2 modes that are based on the presence of the 'dev' file in the root of the project.
 - When 'dev' is present, Cython will be used to compile the .pyx sources. This is the development mode
   (as you get it in the git repository).
 - When 'dev' is absent, C/C++ compiler will be used to compile the .cpp sources (that were prepared in
   in the development mode). This is the distribution mode (as you get it on PyPI).

This way the package can be used without or with an incompatible version of Cython.

The idea comes from: https://github.com/MattShannon/bandmat
"""


print('Development mode: Compiling Cython modules from .pyx sources.')

ext_clipper = Extension("clipper",
                        sources=["clipper/clipper.pyx", "clipper/ClipperLib/clipper.cpp"],
                        language="c++",
                        # define extra macro definitions that are used by clipper
                        # Available definitions that can be used with pyclipper:
                        # use_lines, use_int32
                        # See clipper/ClipperLib/clipper.hpp
                        # define_macros=[('use_lines', 1)]
                        )
ext_clipperx = Extension("clipperx",
                         sources=["clipper/clipperx.pyx", "clipper/ClipperLib/clipper.cpp"],
                         language="c++",
                         include_dirs=[np.get_include()],
                         # define extra macro definitions that are used by clipper
                         # Available definitions that can be used with pyclipper:
                         # use_lines, use_int32
                         # See clipper/ClipperLib/clipper.hpp
                         # define_macros=[('use_lines', 1)]
                         )

with open("README.rst", "r", encoding='utf-8') as readme:
    long_description = readme.read()

setup(
    name='clipper',
    description='Cython wrapper for the C++ translation of the Angus Johnson\'s Clipper library (ver. 6.4.2)',
    long_description=long_description,
    author='Angus Johnson, Maxime Chalton, Lukas Treyer, Gregor Ratajc',
    author_email='me@gregorratajc.com',
    license='MIT',
    url='https://github.com/greginvm/pyclipper',
    keywords=[
        'polygon clipping, polygon intersection, polygon union, polygon offsetting, polygon boolean, polygon, clipping, clipper, vatti'],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "Programming Language :: C++",
        "Environment :: Other Environment",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "License :: OSI Approved",
        "License :: OSI Approved :: MIT License",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    ext_modules=cythonize([ext_clipper, ext_clipperx]),
)
