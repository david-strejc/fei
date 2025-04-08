#!/bin/bash
# Script to test fei's ability to perform a multi-step task with shell commands

# Unset any existing ANTHROPIC_API_KEY environment variable
unset ANTHROPIC_API_KEY

# Set the API key as an environment variable
# IMPORTANT: Replace "YOUR_ANTHROPIC_API_KEY" with your actual key
# or source it from a secure location (e.g., .env file or environment variable)
export ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY"

# Check if the API key is set
if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" == "YOUR_ANTHROPIC_API_KEY" ]; then
  echo "Error: ANTHROPIC_API_KEY is not set or is still the placeholder value."
  echo "Please set your Anthropic API key in test_fei_shell_task.sh before running."
  exit 1
fi

# Create a task file with instructions for fei
cat > fei_task.txt << 'EOF'
I want you to perform a multi-step task that involves creating and manipulating files in a temporary directory. Please follow these steps:

1. Create a temporary directory named "fei_test_dir" in the current directory
2. Create a Python script named "data_processor.py" in the temporary directory with the following functionality:
   - It should read a CSV file named "input.csv"
   - It should process the data (calculate the sum and average of numeric columns)
   - It should write the results to a file named "results.txt"

3. Create a sample CSV file named "input.csv" in the temporary directory with the following content:
   ```
   Name,Age,Score
   Alice,25,95
   Bob,30,85
   Charlie,22,90
   David,28,88
   ```

4. Run the Python script and verify that it works correctly
5. Modify the Python script to also calculate the median of the numeric columns
6. Run the modified script and verify that it works with the new functionality

Please execute each step and show me the results. Make sure to handle any errors that might occur.
EOF

# Run fei with the task
echo "Running fei with the multi-step task..."
python -m fei --task "$(cat fei_task.txt)" --max-iterations 10

# Clean up
rm fei_task.txt
