#!/usr/bin/python3
# -*- coding: utf-8 -*-


from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

extensions = [
    Extension("script", ["script.py"])
]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize( extensions ),
)



