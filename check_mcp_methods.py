#!/usr/bin/env python3
"""
Check available methods in the MCP server
"""

import sys
import json
import asyncio
import subprocess
from pathlib import Path

async def check_mcp_methods():
    """Start an MCP server and list available methods"""
    print("Starting Brave Search MCP server and listing available methods...")
    
    # Start the MCP server
    process = subprocess.Popen(
        ["npx", "-y", "@modelcontextprotocol/server-brave-search"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env={"BRAVE_API_KEY": "BSABGuCvrv8TWsq-MpBTip9bnRi6JUg"}
    )
    
    try:
        # Wait for server to start
        await asyncio.sleep(2)
        
        # Try to get available methods
        methods_to_try = [
            "rpc.discover",
            "rpc.listMethods",
            "rpc.methods",
            "system.listMethods",
            "system.methods",
            "brave-search.methods",
            "brave-search.listMethods"
        ]
        
        for method in methods_to_try:
            print(f"\nTrying to call method: {method}")
            
            # Build request payload
            payload = {
                "jsonrpc": "2.0",
                "method": method,
                "params": {},
                "id": 1
            }
            
            payload_str = json.dumps(payload) + "\n"
            
            # Send request
            process.stdin.write(payload_str)
            process.stdin.flush()
            
            # Read response
            response_str = process.stdout.readline()
            
            try:
                # Parse response
                result = json.loads(response_str)
                print(f"Response: {json.dumps(result, indent=2)}")
            except json.JSONDecodeError:
                print(f"Error parsing response: {response_str}")
        
        # List all services
        print("\nTrying to search for available services...")
        payload = {
            "jsonrpc": "2.0",
            "method": "search",
            "params": {"query": "test"},
            "id": 1
        }
        
        payload_str = json.dumps(payload) + "\n"
        
        # Send request
        process.stdin.write(payload_str)
        process.stdin.flush()
        
        # Read response
        response_str = process.stdout.readline()
        
        try:
            # Parse response
            result = json.loads(response_str)
            print(f"Search response: {json.dumps(result, indent=2)}")
        except json.JSONDecodeError:
            print(f"Error parsing search response: {response_str}")
        
    finally:
        # Stop the server
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()

if __name__ == "__main__":
    asyncio.run(check_mcp_methods())