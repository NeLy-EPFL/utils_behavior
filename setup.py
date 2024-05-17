from setuptools import setup, find_packages

setup(
    name="utils_behavior",
    version="0.1",
    packages=find_packages(),
    description="Utility functions to manipulate and analyze behavior videos and associated data (tracking...) acquired in various setups and settings.",
    author="Matthias Durrieu",
    author_email="matthiasdurrieu@gmail.com",
    url="https://github.com/NeLy-EPFL/utils_behavior",
    install_requires=open("requirements.txt").read().splitlines(),
)
