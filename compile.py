from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_modules = [
    Extension("face_recognition_sdk.sdk",  ["face_recognition_sdk/sdk.py"]),
    # Extension("mymodule2",  ["mymodule2.py"]),
#   ... all your modules that need be compiled ...
]
setup(
    name = 'Insight Face',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)