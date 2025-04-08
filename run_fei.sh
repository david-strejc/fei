#!/bin/bash
# Script to run fei assistant with the correct API key

# Unset any existing ANTHROPIC_API_KEY environment variable
unset ANTHROPIC_API_KEY

# Set the API key as an environment variable
# IMPORTANT: Replace "YOUR_ANTHROPIC_API_KEY" with your actual key
# or source it from a secure location (e.g., .env file or environment variable)
export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY"

# Check if the API key is set
if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" == "YOUR_ANTHROPIC_API_KEY" ]; then
  echo "Error: ANTHROPIC_API_KEY is not set or is still the placeholder value."
  echo "Please set your Anthropic API key in run_fei.sh before running."
  exit 1
fi

# Run fei with any arguments passed to this script
python -m fei "$@"
