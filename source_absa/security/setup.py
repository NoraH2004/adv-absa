from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py
import os
from Cython.Build import cythonize


def get_file_paths(root_dir):
    """get filepaths for compilation"""
    paths = []
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if os.path.splitext(filename)[1] != '.py' or 'test_' in filename:
                continue
            file_path = os.path.join(root, filename)
            paths.append(file_path)
    return paths


# noinspection PyPep8Naming
class build_py(_build_py):

    def build_packages(self):
        pass


setup(
    name='security',
    version='0.2.1',
    packages=find_packages(exclude=['tests', 'test_*']),
    ext_modules=cythonize(
        get_file_paths('security'),
        compiler_directives={'language_level': 3}
    ),
    cmdclass={
        'build_py': build_py
    },
    license='(c) DeepOpinion',
    url='https://github.com/deepopinion',
    author='DeepOpinion Team',
    author_email='info@deepopinion.ai',
    install_requires=[
        "numpy~=1.17.4",
        "cryptography~=2.9",
        "pyjwt~=1.7.1",
        "flask~=1.0.2"
    ],
)
