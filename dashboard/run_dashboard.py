#!/usr/bin/env python3
"""
LLM Alignment Evaluator Dashboard Runner

This script can launch either the main dashboard or the constitution editor,
or both simultaneously on different ports.
"""

import argparse
import subprocess
import os
import sys
from pathlib import Path

def run_app(app_path, port=None):
    """Run a Streamlit app at the specified path."""
    cmd = ["streamlit", "run", app_path]
    if port:
        cmd.extend(["--server.port", str(port)])
    
    try:
        process = subprocess.Popen(cmd)
        print(f"Started app at {app_path} on port {port if port else '8501'}")
        return process
    except Exception as e:
        print(f"Error starting app: {str(e)}")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run LLM Alignment Evaluator Dashboard")
    parser.add_argument(
        "--app", 
        choices=["dashboard", "constitution", "both"],
        default="dashboard",
        help="Which app to run: main dashboard, constitution editor, or both"
    )
    parser.add_argument(
        "--dashboard-port", 
        type=int, 
        default=8501,
        help="Port to run the main dashboard on"
    )
    parser.add_argument(
        "--constitution-port", 
        type=int, 
        default=8502,
        help="Port to run the constitution editor on"
    )
    
    args = parser.parse_args()
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Define paths to the apps
    dashboard_path = script_dir / "streamlit_app.py"
    constitution_path = script_dir / "constitution_editor.py"
    
    # Ensure the app files exist
    if not dashboard_path.exists():
        print(f"Error: Main dashboard file not found at {dashboard_path}")
        sys.exit(1)
    
    if (args.app in ["constitution", "both"]) and not constitution_path.exists():
        print(f"Error: Constitution editor file not found at {constitution_path}")
        sys.exit(1)
    
    # Launch the requested app(s)
    dashboard_process = None
    constitution_process = None
    
    try:
        if args.app in ["dashboard", "both"]:
            dashboard_process = run_app(dashboard_path, args.dashboard_port)
            print(f"Main dashboard running at http://localhost:{args.dashboard_port}")
        
        if args.app in ["constitution", "both"]:
            constitution_process = run_app(constitution_path, args.constitution_port)
            print(f"Constitution editor running at http://localhost:{args.constitution_port}")
        
        # Keep the script running until interrupted
        print("\nPress Ctrl+C to stop the dashboard(s)...\n")
        
        # Wait for the process(es) to complete (which won't happen unless they crash)
        if dashboard_process:
            dashboard_process.wait()
        if constitution_process:
            constitution_process.wait()
    
    except KeyboardInterrupt:
        print("\nShutting down dashboards...")
        # Clean up processes on keyboard interrupt
        if dashboard_process:
            dashboard_process.terminate()
        if constitution_process:
            constitution_process.terminate()
    
    print("Dashboard(s) stopped.")

if __name__ == "__main__":
    main() 