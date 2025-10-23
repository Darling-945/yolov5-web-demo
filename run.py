#!/usr/bin/env python3
"""
YOLO Web Demo - Main Application Runner
"""
import os
import sys
from app import app

def main():
    """Main function to run the Flask application"""
    print("Starting YOLO Web Demo...")
    print("Application will be available at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")

    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()