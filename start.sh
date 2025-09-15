#!/bin/bash

# Ploop AI Server Startup Script

echo "Starting Ploop AI Server..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file with your configuration before running the server"
fi

# Start the server
echo "Starting server on http://localhost:8000"
echo "API Documentation available at http://localhost:8000/docs"
python main.py