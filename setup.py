from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="meta_metrics",
    version="0.1.0",
    author="Genta Indra Winata",
    author_email="gentaindrawinata@gmail.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache 2.0 License",
    url="https://github.com/gentaiscool/meta-metrics",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    project_urls={
        "Bug Tracker": "https://github.com/gentaiscool/meta-metrics/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0 License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "bayesian-optimization",
        "evaluate==0.4.0",
        "scipy",
        "transformers==4.42.3",
        "sentencepiece",
        "sacrebleu==2.4.2",
        "unbabel-comet==2.2.2",
        "pandas",
        "numpy",
        "tensorflow",
        "torch",
        "torchvision",
        "tf-slim>=1.1",
        "bert_score",
        "gdown"
    ],
    python_requires=">=3.10",
)
