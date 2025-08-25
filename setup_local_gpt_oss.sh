#!/bin/bash

echo "🚀 Setting up Local gpt-oss-20b Environment..."

# Check if we're in the right directory
if [ ! -f "src/local_red_teaming.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Install local model dependencies
echo "📦 Installing local model dependencies..."
pip install transformers torch accelerate

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected - will use CUDA"
    echo "💡 Make sure you have enough VRAM (at least 16GB recommended for gpt-oss-20b)"
else
    echo "⚠️  No NVIDIA GPU detected - will use CPU"
    echo "💡 This will be much slower. Consider using Google Colab or a cloud GPU"
fi

# Check available memory
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    total_mem=$(sysctl -n hw.memsize | awk '{print $0/1024/1024/1024}')
    echo "💾 Available RAM: ${total_mem} GB"
    if (( $(echo "$total_mem < 32" | bc -l) )); then
        echo "⚠️  Warning: Less than 32GB RAM available. gpt-oss-20b may not fit."
        echo "💡 Consider using quantization or smaller models"
    fi
else
    # Linux
    total_mem=$(free -g | awk 'NR==2{print $2}')
    echo "💾 Available RAM: ${total_mem} GB"
    if [ "$total_mem" -lt 32 ]; then
        echo "⚠️  Warning: Less than 32GB RAM available. gpt-oss-20b may not fit."
        echo "💡 Consider using quantization or smaller models"
    fi
fi

# Create results directory
echo "📁 Creating results directory..."
mkdir -p results/local_red_teaming

echo ""
echo "🎯 Setup Complete! Next steps:"
echo "1. Run: python src/local_red_teaming.py"
echo "2. The script will automatically download gpt-oss-20b (~40GB)"
echo "3. Check results in: results/local_red_teaming/"
echo ""
echo "🚨 Important Notes:"
echo "   - First run will download the model (this may take a while)"
echo "   - You need at least 32GB RAM/VRAM for full precision"
echo "   - Consider using 8-bit quantization if memory is limited"
echo "   - Results will be saved as findings.json files for Kaggle submission"
echo ""
echo "💡 Alternative: Use Google Colab if you don't have enough local resources"
echo "   - Colab provides free GPU access"
echo "   - Modify the script to work with Colab's environment" 