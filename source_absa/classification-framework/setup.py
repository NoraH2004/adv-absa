from setuptools import find_packages, setup

setup(
    name='clsframework',
    version='0.3.5',
    packages=find_packages(),
    include_package_data=True,
    license='(c) DeepOpinion',
    url='https://github.com/deepopinion/classification_framework',
    author='DeepOpinion',
    author_email='hello@deepopinion.ai',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],

    install_requires=[
        "torch",         # pyTorch, deep Learning library
        "transformers",         # Library for transformer models in pyTorch (BERT, RoBERTa, ...)
        "tensorboard",          # Logging of metrics, values, images and webfrontend for visualization
        "numpy",                # General Vector/Matrix support
        "tqdm",                 # Progress bar
        "pynvml",               # To get free/used GPU memory etc.
        "security~=0.2.0"       # For model encryption
    ],
    zip_safe=False,
)
