import os
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="metametrics",
    version="0.0.1",
    author="Genta Indra Winata",
    author_email="gentaindrawinata@gmail.com",
    description="MetaMetrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache 2.0 License",
    url="https://github.com/meta-metrics/meta-metrics",
    project_urls={
        "Bug Tracker": "https://github.com/meta-metrics/meta-metrics/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
    ],
    packages = ['metametrics'],
    python_requires=">=3.10",
)
