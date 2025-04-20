#!/usr/bin/env python3
"""
Setup script for memdir-tools
"""

from setuptools import setup, find_packages

setup(
    name="memdir-tools",
    version="0.2.0",
    description="Maildir-inspired hierarchical memory management system with HTTP API",
    author="Claude",
    author_email="claude@anthropic.com",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "memdir=memdir_tools.__main__:main",
            "memdir-server=memdir_tools.run_server:main",
        ],
    },
    install_requires=[
        "python-dateutil",  # For date parsing
    ],
    extras_require={
        "server": [
            "flask>=2.0.0",
            "werkzeug>=2.0.0",
            "requests>=2.25.0",
        ],
        "client": [
            "requests>=2.25.0",
        ],
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)