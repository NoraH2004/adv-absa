from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py
import os
from Cython.Build import cythonize


def get_ext_paths(root_dir):
    """get filepaths for compilation"""
    paths = []

    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if os.path.splitext(filename)[1] != '.py':
                continue
            file_path = os.path.join(root, filename)
            paths.append(file_path)
    return paths


# noinspection PyPep8Naming
class build_py(_build_py):

    def build_packages(self):
        pass


setup(
    name='nlp',
    version='0.4.1',
    packages=find_packages(),
    ext_modules=cythonize(
        get_ext_paths('nlp'),
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
        "langdetect==1.0.7",
        "spacy~=2.2.4",
        "requests~=2.22.0"
    ],
)
