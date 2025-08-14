#!/usr/bin/env python3
"""
RML-AI: Resonant Memory Learning
A New Generation of AI for Mission-Critical Applications
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rml-ai",
    version="0.1.0",
    author="RML AI Team",
    author_email="team@rml-ai.com",
    description="Resonant Memory Learning - A New Generation of AI for Mission-Critical Applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/rml-ai",
    project_urls={
        "Bug Reports": "https://github.com/your-username/rml-ai/issues",
        "Source": "https://github.com/your-username/rml-ai",
        "Documentation": "https://docs.rml-ai.com",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rml-ai=rml_ai.cli:main",
            "rml-server=rml_ai.server:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="ai, machine-learning, nlp, transformers, rml, resonant-memory",
) 