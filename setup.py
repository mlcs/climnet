"""setup.py for climnet."""
from setuptools import find_packages
from setuptools import setup

setup(
    name='climnet',
    version='1.0',
    description=('Library for research on climate networks.'),
    author='Felix Strnad, Jakob Schlör',
    author_email='jakob.schloer@uni-tuebingen.de',
    packages=find_packages()
)
