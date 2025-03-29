#!/bin/bash

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p results
mkdir -p prompts

# Copy example environment file
echo "OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here" > .env.example

echo "Setup complete! Please:"
echo "1. Copy .env.example to .env"
echo "2. Add your API keys to .env"
echo "3. Run 'python src/main.py' to start evaluation" 