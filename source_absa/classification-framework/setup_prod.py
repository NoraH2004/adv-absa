from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py
import os
from Cython.Build import cythonize
import pkg_resources


def get_ext_paths(root_dir):
    """get filepaths for compilation"""
    paths = []
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if os.path.splitext(filename)[1] not in ['.py', '.json'] or 'test' in root or 'template' in filename:
                continue
            file_path = os.path.join(root, filename)
            paths.append(file_path)
    return paths


# noinspection PyPep8Naming
class build_py(_build_py):

    def build_packages(self):
        pass


setup(
    name='clsframework',
    version=pkg_resources.get_distribution('clsframework').version,
    packages=find_packages(exclude=['tests', 'test_*']),
    package_data={'clsframework': ['modules/*.json']},
    ext_modules=cythonize(
        get_ext_paths('clsframework'),
        compiler_directives={'language_level': 3}
    ),
    license='(c) DeepOpinion',
    cmdclass={
        'build_py': build_py
    },
    url='https://github.com/deepopinion/classification_framework',
    author='DeepOpinion',
    author_email='hello@deepopinion.ai',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch",  # pyTorch, deep Learning library
        "transformers",  # Library for transformer models in pyTorch (BERT, RoBERTa, ...)
        "tensorboard",  # Logging of metrics, values, images and webfrontend for visualization
        "numpy",  # General Vector/Matrix support
        "tqdm",  # Progress bar
        "pynvml",  # To get free/used GPU memory etc.
        "security~=0.2.0",  # For model encryption
    ]
)
