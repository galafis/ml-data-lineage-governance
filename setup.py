"""Package setup for ml-data-lineage-governance."""

from setuptools import find_packages, setup

setup(
    name="ml-data-lineage-governance",
    version="1.0.0",
    author="Gabriel Demetrios Lafis",
    author_email="gabriel.lafis@gmail.com",
    description="Data lineage tracking and governance platform for ML pipelines",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
)
