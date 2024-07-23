import subprocess
import os
import logging
from setuptools import find_packages, setup
from setuptools.command.install import install

# Setup logging
logging.basicConfig(level=logging.INFO)

# Define a setup command
class SetupInstallCommand(install):
    description = 'Run additional shell commands for setup'
    user_options = []

    def run(self):
        # Navigate to the meta_metrics/metrics directory
        os.chdir('meta_metrics/metrics')

        # Clone BLEURT repository if it doesn't exist
        if not os.path.isdir('bleurt'):
            logging.info("Cloning BLEURT repository ...")
            subprocess.run(["git", "clone", "https://github.com/google-research/bleurt.git"])
            
            # Navigate to the bleurt directory
            os.chdir('bleurt')

            # Install BLEURT
            logging.info("Installing BLEURT ...")
            result = subprocess.run(["pip", "install", "."], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                logging.info("BLEURT installed successfully.")
            else:
                logging.error("Failed to install BLEURT.")
                logging.error(result.stderr.decode())
                return
        else:
            logging.info("Skipping BLEURT installation as BLEURT already exists.")

        # Run the standard install process
        install.run(self)

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
        "numpy==1.26.4",
        "tensorflow==2.16.2",
        "torch==2.3.1",
        "torchvision",
        "tf-slim>=1.1",
        "bert_score",
        'regex',
        'six',
        "gdown",
        "huggingface-hub",
        "scikit-learn",
        # GEMBA Requirements
        "openai>=1.0.0",
        "termcolor",
        "pexpect",
        "ipdb",
        "absl-py"
    ],
    python_requires=">=3.10",
    cmdclass={
        'install': SetupInstallCommand,
    },
)
