#!/bin/bash
# Script to test if the API key from the .env file is working

# Unset any existing ANTHROPIC_API_KEY environment variable
unset ANTHROPIC_API_KEY

# Print the current environment variable
echo "Before loading .env: ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY"

# Create a simple Python script to load the .env file and print the API key
cat > test_env.py << 'EOF'
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv(override=True)

# Print the API key
print(f"After loading .env: ANTHROPIC_API_KEY={os.environ.get('ANTHROPIC_API_KEY')}")
EOF

# Run the Python script
python test_env.py

# Clean up
rm test_env.py

# Try running fei with the API key from the .env file
echo -e "\nTesting fei with API key from .env file..."
python -m fei --message "This is a test message to verify the API key from the .env file is working."
