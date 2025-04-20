#!/usr/bin/env python3
"""
Setup script for Fei code assistant
"""

from setuptools import setup, find_packages

# Read the content of README.md
with open("docs/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fei",
    version="0.1.0",
    author="Claude AI",
    author_email="noreply@anthropic.com",
    description="Advanced code assistant with universal AI tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fei",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "anthropic>=0.49.0",
        "litellm>=1.26.0",
        "requests>=2.32.0",
        "rich>=13.9.0",
        "click>=8.1.0",
        "pydantic>=2.10.0",
        "textual>=0.47.1",
        "tree-sitter-languages>=1.8.0",
    ],
    entry_points={
        "console_scripts": [
            "fei=fei.__main__:main",
        ],
    },
)