from setuptools import find_packages
from distutils.core import setup

setup(
    name="aerial_gym",
    version="2.0.0",
    author="",
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email="",
    description="Isaac Gym environments for Aerial Robots (SHVD)",
    install_requires=[
        "isaacgym",
        "matplotlib",
        "numpy",
        "torch",
        "pytorch3d",
        "warp-lang==1.0.0",
        "trimesh",
        "urdfpy",
        "numpy==1.23",
        "gymnasium",
        "rl-games",
        "sample-factory",
    ],
)
