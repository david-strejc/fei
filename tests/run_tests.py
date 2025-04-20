#!/usr/bin/env python3
"""
Run all tests for the Fei code assistant.
"""

import os
import sys
import pytest


def main():
    """Run all tests."""
    # Add project root to path
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    
    # Run tests
    args = [
        "-v",  # Verbose output
        "--cov=fei",  # Coverage for fei package
        "--cov-report=term",  # Coverage report in terminal
        "--cov-report=html:coverage_html",  # HTML coverage report
    ]
    
    # Add test directories
    args.extend([
        "tests/providers",
        "tests/memory",
        "tests/evolution",
        "tests/mcp",
        "tests/ui",
        "tests/tools",
        "tests/core",
        "tests/utils"
    ])
    
    # Run pytest
    return pytest.main(args)


if __name__ == "__main__":
    sys.exit(main())
