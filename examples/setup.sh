#!/bin/bash

# Enumeraite Setup Script
# This script helps you set up Enumeraite quickly

set -e  # Exit on any error

echo "🔍 Enumeraite Setup Script"
echo "========================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "   Please install Python 3.9 or later and try again."
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
required_version="3.9"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,9) else 1)" 2>/dev/null; then
    echo "❌ Python $required_version or later is required."
    echo "   Current version: $python_version"
    exit 1
fi

echo "✅ Python $python_version found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install Enumeraite
echo "📥 Installing Enumeraite..."
pip install -e .

echo "✅ Installation complete!"

# Check if configuration exists
config_dir="$HOME/.config/enumeraite"
config_file="$config_dir/config.json"

if [ ! -f "$config_file" ]; then
    echo ""
    echo "⚙️  Configuration Setup"
    echo "====================="
    echo "No configuration file found. Let's create one."

    # Create config directory
    mkdir -p "$config_dir"

    # Ask for API keys
    read -p "🔑 Enter your OpenAI API key (or press Enter to skip): " openai_key
    read -p "🔑 Enter your Claude API key (or press Enter to skip): " claude_key

    # Create basic config
    cat > "$config_file" << EOF
{
  "providers": {
EOF

    if [ ! -z "$openai_key" ]; then
        cat >> "$config_file" << EOF
    "openai": {
      "api_key": "$openai_key",
      "model": "gpt-4",
      "max_tokens": 1000
    }EOF
        if [ ! -z "$claude_key" ]; then
            echo "," >> "$config_file"
        fi
        echo "" >> "$config_file"
    fi

    if [ ! -z "$claude_key" ]; then
        cat >> "$config_file" << EOF
    "claude": {
      "api_key": "$claude_key",
      "model": "anthropic/claude-sonnet-4",
      "max_tokens": 1000
    }
EOF
    fi

    cat >> "$config_file" << EOF
  },
  "default_provider": "openai",
  "http_validator": {
    "timeout": 10,
    "max_concurrent": 20
  }
}
EOF

    echo "✅ Configuration file created at $config_file"
else
    echo "✅ Configuration file already exists at $config_file"
fi

# Test installation
echo ""
echo "🧪 Testing Installation"
echo "======================"

if enumeraite --help > /dev/null 2>&1; then
    echo "✅ CLI is working correctly"
else
    echo "❌ CLI test failed"
    exit 1
fi

# Run tests if available
if [ -d "tests" ]; then
    echo "🔍 Running tests..."
    if python -m pytest tests/ -q; then
        echo "✅ All tests passed"
    else
        echo "⚠️  Some tests failed (this might be expected without API keys)"
    fi
fi

# Create sample files
echo ""
echo "📄 Creating Sample Files"
echo "======================"

if [ ! -f "sample_paths.txt" ]; then
    cp examples/sample_paths.txt sample_paths.txt
    echo "✅ Created sample_paths.txt"
fi

echo ""
echo "🎉 Setup Complete!"
echo "================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Try the CLI: enumeraite batch --target example.com --input sample_paths.txt --count 5"
echo "3. For continuous mode: enumeraite continuous --target example.com --input sample_paths.txt --duration 5m"
echo ""
echo "📚 Documentation:"
echo "   - README.md - Main documentation"
echo "   - docs/CONFIGURATION.md - Configuration guide"
echo "   - examples/ - Usage examples"
echo ""
echo "🔧 Configuration:"
echo "   - Config file: $config_file"
echo "   - Environment variables: OPENAI_API_KEY, ANTHROPIC_API_KEY"
echo ""

if [ -z "$openai_key" ] && [ -z "$claude_key" ]; then
    echo "⚠️  Warning: No API keys configured."
    echo "   Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables,"
    echo "   or edit the config file at $config_file"
fi

echo "Happy enumerating! 🔍"