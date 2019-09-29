import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

# Python interface
setup(
    name='arcsim',
    install_requires=['torch'],
    ext_modules=[
        CppExtension(
            name='arcsim',
            include_dirs=['./arcsim/src/','./arcsim/dependencies/include'],
            sources=[
                'pybind/bind.cpp',
            ],
            libraries=['make_pytorch','json','taucs','alglib',
            'png','z','lapack','blas','boost_system-mt','boost_filesystem-mt','boost_thread-mt','gomp','glut','GLU','GL','glapi','GLdispatch'],
            library_dirs=['objs','./arcsim/dependencies/lib'],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
