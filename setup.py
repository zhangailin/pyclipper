import numpy as np
import os

from distutils.command.build_ext import build_ext
try:
    from setuptools import Extension, setup
except ImportError:
    from distutils.core import Extension, setup

from Cython.Build import cythonize


# By subclassing build_extensions we have the actual compiler that will be used which is really known only after finalize_options
# http://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
compile_options =  {'msvc'  : ['/Ox', '/EHsc'],
                    'other' : ['-O3', '-Wno-strict-prototypes', '-Wno-unused-function']}
link_options    =  {'msvc'  : [],
                    'other' : []}

class build_ext_options:
    def build_options(self):
        for e in self.extensions:
            e.extra_compile_args = compile_options.get(
                self.compiler.compiler_type, compile_options['other'])
        for e in self.extensions:
            e.extra_link_args = link_options.get(
                self.compiler.compiler_type, link_options['other'])

class build_ext_subclass(build_ext, build_ext_options):
    def build_extensions(self):
        build_ext_options.build_options(self)
        build_ext.build_extensions(self)


info = {}
filename = os.path.join('clipper', '_version.py')
exec(compile(open(filename, 'rb').read().replace(b'\r\n', b'\n'),
                 filename, 'exec'), info)
VERSION = info['__version__']


print("Compiling Cython modules from .pyx sources.")

exts = []
for mod_name in ["clipper", "clipperx"]:
    exts.append(Extension("clipper.{}".format(mod_name),
        sources=["clipper/{}.pyx".format(mod_name),
            "clipper/ClipperLib/clipper.cpp"],
        language="c++",
        include_dirs=[np.get_include()],
        # define extra macro definitions that are used by clipper
        # Available definitions that can be used with pyclipper:
        # use_lines, use_int32
        # See clipper/ClipperLib/clipper.hpp
        # define_macros=[('use_lines', 1)]
        ))

with open("README.rst", "r", encoding='utf-8') as readme:
    long_description = readme.read()

setup(
    name='clipper',
    version=VERSION,
    packages=['clipper'],
    description='Cython wrapper for the C++ translation of the Angus Johnson\'s Clipper library (ver. 6.4.2)',
    long_description=long_description,
    author='Angus Johnson, Maxime Chalton, Lukas Treyer, Gregor Ratajc',
    license='MIT',
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
    ext_modules=cythonize(exts),
    cmdclass = {'build_ext': build_ext_subclass},
)
