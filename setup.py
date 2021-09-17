import os

from setuptools import setup, find_packages

globalVersion = {}
with open(os.path.join('data_services', 'version.py')) as fp:
    exec(fp.read(), globalVersion)

setup(
    name="data_services",
    version=globalVersion['__version__'],
    package_dir={'': '.'},
    packages=find_packages(exclude=["tests"]),
    scripts=[],
    include_package_data=True,
    zip_safe=False
)
