import pathlib

import pkg_resources
import setuptools

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="DenseSense",
    version="0.1",
    author="Axel Wickman",
    author_email="axelwickm@gmail.com",
    description="Integration of Facebook's DensePose algorithm with person tracking and clothing recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    url="https://github.com/Axelwickm/DenseSense",
    packages=["DenseSense", "DenseSense.algorithms", "DenseSense.utils"],
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
