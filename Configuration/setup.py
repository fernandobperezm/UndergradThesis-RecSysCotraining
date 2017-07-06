import numpy
from setuptools import Command, Extension, setup
from Cython.Build import cythonize

from Cython.Compiler.Options import _directive_defaults

_directive_defaults['linetrace'] = True
_directive_defaults['binding'] = True

extensions = [
    Extension(name='implementation._cython._similarity',
              sources=["implementation/_cython/_similarity.pyx"], define_macros=[('CYTHON_TRACE', '1')]),
    Extension(name='implementation._cython._mf',
              sources=["implementation/_cython/_mf.pyx"], define_macros=[('CYTHON_TRACE', '1')]),
]

setup(
    name='Co-Training',
    version="0.1",
    description='Co-training with Recommender Systems framework.',
    url='https://github.com/fernandobperezm/recsys-cotraining',
    author='Fernando Benjamín Pérez Maurera',
    author_email='fernandobenjamin.perez@polimi.it',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Alpha',
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Cython',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
        ],
    install_requires=['numpy', 'scipy>=0.16'],
    packages=['implementation', 'implementation.recommenders', 'implementation._cython', 'implementation.utils'],
    setup_requires=["Cython >= 0.19"],
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)
