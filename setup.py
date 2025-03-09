import os
from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open(os.path.join(os.path.dirname(__file__), "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="utilz",
    version="0.1.0",
    description="A collection of utility functions and modules.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Usman Bashir",
    url="https://github.com/drusmanbashir/utilz",
    packages=find_packages(),  # This will find the inner "utilz" package automatically.
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
