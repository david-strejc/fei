#!/usr/bin/env python3
"""
Test real application scenario with keys from .env
"""

import os
import tempfile
import shutil
import subprocess
from pathlib import Path

def main():
    print("Testing real application scenario with .env files")
    print("-" * 50)
    
    # Save original environment
    original_env = {}
    for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "BRAVE_API_KEY", "LLM_API_KEY"]:
        if key in os.environ:
            original_env[key] = os.environ[key]
            del os.environ[key]
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_env_path = os.path.join(temp_dir, '.env')
    
    try:
        # Create a custom .env file in the temp directory
        with open(temp_env_path, 'w') as f:
            f.write("ANTHROPIC_API_KEY=test_anthropic_key_for_app\n")
            f.write("OPENAI_API_KEY=test_openai_key_for_app\n")
        
        # Create a simple script to test
        test_script_path = os.path.join(temp_dir, 'test_script.py')
        with open(test_script_path, 'w') as f:
            f.write("""#!/usr/bin/env python
import os
from fei.utils.config import get_config

# Get config instance
config = get_config()

# Test API keys
providers = ["anthropic", "openai", "groq"]
for provider in providers:
    key = f"{provider}.api_key"
    value = config.get(key)
    print(f"{key}: {value}")
""")
        
        # Run the script in the temp directory
        print("Running test in temporary directory with custom .env file:")
        result = subprocess.run(
            ['python', test_script_path], 
            cwd=temp_dir,
            capture_output=True, 
            text=True
        )
        
        print(result.stdout)
        
        # Now set an environment variable directly and run again
        os.environ["ANTHROPIC_API_KEY"] = "direct_env_anthropic_key"
        
        print("Running test with direct environment variable:")
        result = subprocess.run(
            ['python', test_script_path], 
            cwd=temp_dir,
            capture_output=True, 
            text=True
        )
        
        print(result.stdout)
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        
        # Restore original environment
        for key, value in original_env.items():
            os.environ[key] = value
    
    print("-" * 50)
    print("Real application scenario test completed")

if __name__ == "__main__":
    main()