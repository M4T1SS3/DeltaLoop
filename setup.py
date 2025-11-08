"""Setup configuration for DeltaLoop."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="deltaloop",
    version="0.1.0",
    author="DeltaLoop Contributors",
    author_email="markorester@gmail.com",
    description="Stop optimizing prompts. Start optimizing models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deltaloop/deltaloop",
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "peft>=0.7.0",
        "trl>=0.7.0",
        "bitsandbytes>=0.41.0",
        "accelerate>=0.25.0",
        "click>=8.0",
        "pydantic>=2.0",
        "rich>=13.0",
    ],
    extras_require={
        "unsloth": ["unsloth[colab-new]>=2024.1"],
        "axolotl": ["axolotl>=0.4.0"],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "ruff>=0.1.0",
            "mypy>=1.0",
        ],
        "all": [
            "unsloth[colab-new]>=2024.1",
            "axolotl>=0.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "deltaloop=deltaloop.cli:cli",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
