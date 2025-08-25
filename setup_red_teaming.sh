#!/bin/bash

echo "🚀 Setting up Red-Teaming Challenge Environment..."

# Check if we're in the right directory
if [ ! -f "src/red_teaming_challenge.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Create new branch for the challenge
echo "📝 Creating new branch: red-teaming-challenge"
git checkout -b red-teaming-challenge

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set"
    echo "Please set your OpenAI API key:"
    echo "export OPENAI_API_KEY=sk-your-key-here"
    echo ""
    echo "Or create a .env file with:"
    echo "OPENAI_API_KEY=sk-your-key-here"
else
    echo "✅ OpenAI API key is set"
fi

# Create results directory
echo "📁 Creating results directory..."
mkdir -p results/red_teaming_challenge

# Test the setup
echo "🧪 Testing setup..."
python src/red_teaming_challenge.py --help 2>/dev/null || echo "⚠️  Script test failed - this is normal for first run"

echo ""
echo "🎯 Setup Complete! Next steps:"
echo "1. Set your OpenAI API key (if not already set)"
echo "2. Run: python src/red_teaming_challenge.py"
echo "3. Check results in: results/red_teaming_challenge/"
echo ""
echo "📚 Available vulnerability types:"
echo "   - Chain of Thought manipulation"
echo "   - Data Exfiltration"
echo "   - Deceptive Alignment"
echo "   - Evaluation Awareness"
echo "   - Reward Hacking"
echo ""
echo "🚨 Remember: You have ~24 hours to submit to Kaggle!" 