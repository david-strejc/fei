#!/usr/bin/env python3
"""
Script to start the Memdir HTTP API server.
Provides a convenient way to start the server with custom settings.
"""

import os
import sys
import argparse
import uuid
import logging
from memdir_tools.server import app

def configure_logging(debug=False):
    """Configure logging for the server"""
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Memdir HTTP API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--generate-key", action="store_true", help="Generate a new random API key")
    parser.add_argument("--api-key", help="Set a specific API key (overrides MEMDIR_API_KEY)")
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(args.debug)
    
    # Handle API key
    if args.generate_key:
        api_key = str(uuid.uuid4())
        print(f"Generated new API key: {api_key}")
        print("To use this key, run:")
        print(f"export MEMDIR_API_KEY=\"{api_key}\"")
        print("Or provide it when starting the server:")
        print(f"python -m memdir_tools.run_server --api-key \"{api_key}\"")
        os.environ["MEMDIR_API_KEY"] = api_key
    elif args.api_key:
        os.environ["MEMDIR_API_KEY"] = args.api_key
    
    # Check if API key is set
    api_key = os.environ.get("MEMDIR_API_KEY", "")
    if not api_key:
        print("WARNING: No API key set. Using an empty API key is insecure.")
        print("Set an API key using the MEMDIR_API_KEY environment variable or --api-key parameter.")
    
    # Start the server
    print(f"Starting Memdir HTTP API server on {args.host}:{args.port}...")
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()