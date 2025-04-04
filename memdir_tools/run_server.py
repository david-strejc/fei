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
# from memdir_tools.server import app # Remove top-level import

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
    parser.add_argument("--data-dir", help="Set the Memdir data directory (overrides MEMDIR_DATA_DIR env var)")

    args = parser.parse_args()

    # Determine data directory path first
    data_dir = args.data_dir or os.environ.get("MEMDIR_DATA_DIR") or os.path.join(os.getcwd(), "Memdir")
    print(f"Configuring server to use data directory: {data_dir}")
    app.config['MEMDIR_DATA_DIR'] = data_dir # Set directly in Flask config

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
        api_key = "" # Use empty string if none provided, server decorator will handle default

    # Set the API key in the Flask app config
    # Now import the app, *after* config is set
    from memdir_tools.server import app
    app.config['MEMDIR_API_KEY'] = api_key

    # Start the server
    print(f"Starting Memdir HTTP API server on {args.host}:{args.port}...")
    # Pass use_reloader=False to prevent Flask from restarting the process,
    # which can interfere with how the test fixture manages the server process.
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)

if __name__ == "__main__":
    main()
