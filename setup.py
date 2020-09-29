import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DenseSense-Axelwickm",
    version="0.1",
    author="Axel Wickman",
    author_email="axelwickm@gmail.com",
    description="Integration of Facebook's DensePose algorithm with person tracking and clothing recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Axelwickm/DenseSense",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
