# Enumeraite Quick Start Guide

This guide will get you up and running with Enumeraite in 5 minutes.

## Prerequisites

- Python 3.9 or later
- OpenAI API key or Anthropic API key
- Internet connection for validation

## 1. Installation

### Option A: Automated Setup (Recommended)
```bash
git clone https://github.com/your-username/enumeraite.git
cd enumeraite
./examples/setup.sh
```

### Option B: Manual Setup
```bash
git clone https://github.com/your-username/enumeraite.git
cd enumeraite
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

## 2. Configuration

### Option A: Environment Variables (Quick)
```bash
export OPENAI_API_KEY="your-openai-api-key"
# or
export ANTHROPIC_API_KEY="your-claude-api-key"
```

### Option B: Configuration File (Persistent)
```bash
mkdir -p ~/.config/enumeraite
cat > ~/.config/enumeraite/config.json << 'EOF'
{
  "providers": {
    "openai": {
      "api_key": "your-openai-api-key",
      "model": "gpt-4"
    }
  },
  "default_provider": "openai"
}
EOF
```

## 3. First Run

### Create Input File
```bash
cat > paths.txt << 'EOF'
/api/users
/api/auth/login
/admin/dashboard
EOF
```

### Generate Paths
```bash
# Basic generation
enumeraite batch --target example.com --input paths.txt --count 10

# With validation
enumeraite batch --target httpbin.org --input paths.txt --count 10 --validate
```

### Example Output
```
Generating 10 paths for example.com using openai
Generated 10 paths
Validated 10 paths, 3 appear to exist

/api/users/profile
/api/users/settings
/api/auth/logout
/api/auth/refresh
/admin/users
/admin/settings
/admin/reports
/api/data
/api/search
/health
```

## 4. Advanced Usage

### Continuous Discovery
```bash
# Run for 5 minutes
enumeraite continuous --target example.com --input paths.txt --duration 5m

# With custom settings
enumeraite continuous --target example.com --input paths.txt \
  --duration 30m --batch-size 20 --max-empty-rounds 5
```

### JSON Output
```bash
# Save detailed results
enumeraite batch --target example.com --input paths.txt --count 20 \
  --output results.json --advanced --validate
```

### Multiple HTTP Methods
```bash
# Test GET, POST, and PUT
enumeraite batch --target example.com --input paths.txt --count 20 \
  --validate --methods "GET,POST,PUT"
```

### Different AI Provider
```bash
# Use Claude instead of OpenAI
enumeraite batch --target example.com --input paths.txt --count 20 \
  --provider claude
```

## 5. Common Use Cases

### Bug Bounty Reconnaissance
```bash
# Generate and validate endpoints
enumeraite batch --target target.com --input known_paths.txt \
  --count 100 --validate --output discovered.json --advanced

# Continuous discovery during reconnaissance
enumeraite continuous --target target.com --input known_paths.txt \
  --duration 2h --batch-size 30
```

### Penetration Testing
```bash
# Quick endpoint discovery
enumeraite batch --target internal.company.com --input api_paths.txt \
  --count 50 --validate --methods "GET,POST,PUT,DELETE"

# Live discovery with real-time updates
enumeraite continuous --target internal.company.com --input api_paths.txt \
  --duration 1h --max-empty-rounds 10
```

### API Security Assessment
```bash
# Comprehensive endpoint mapping
enumeraite batch --target api.service.com --input base_endpoints.txt \
  --count 200 --validate --provider claude --output assessment.json --advanced
```

## 6. Tips and Best Practices

### Input File Quality
- Use diverse, representative paths as seeds
- Include different API patterns (REST, GraphQL, etc.)
- Add comments to organize your paths
- Update the file with discovered paths

### Provider Selection
- **GPT-4**: Best quality, higher cost
- **GPT-3.5-turbo**: Good balance, lower cost
- **Claude**: Alternative perspective, sometimes better for specific patterns

### Validation Settings
- Start with `--validate` to see what exists
- Use `--methods "GET"` for read-only testing
- Increase `--max-concurrent` for faster validation (be careful with rate limits)

### Continuous Mode
- Use shorter durations first to test effectiveness
- Monitor `--max-empty-rounds` to avoid infinite loops
- The input file is updated in real-time with discoveries

## 7. Troubleshooting

### API Key Issues
```bash
# Check if key is recognized
enumeraite batch --target example.com --input paths.txt --count 1
```

If you see "Configuration error" or "Provider error", check your API keys.

### Rate Limiting
If you hit rate limits:
```bash
# Reduce concurrent requests
enumeraite batch --target example.com --input paths.txt --count 10 \
  --validate --max-concurrent 5

# Use smaller batch sizes in continuous mode
enumeraite continuous --target example.com --input paths.txt \
  --duration 30m --batch-size 10
```

### Validation Timeouts
```bash
# Increase timeout for slow targets
export ENUMERAITE_HTTP_TIMEOUT=30
enumeraite batch --target slow-server.com --input paths.txt --validate
```

### Debug Mode
```bash
export ENUMERAITE_LOG_LEVEL=DEBUG
enumeraite batch --target example.com --input paths.txt --count 5
```

## 8. Next Steps

- Read the full [README.md](../README.md) for complete features
- Check [CONFIGURATION.md](CONFIGURATION.md) for advanced configuration
- Explore [examples/](../examples/) for more usage patterns
- Run the test suite: `pytest tests/`

## 9. Security Reminder

⚠️ **Always ensure you have permission to test your targets!**

Enumeraite is a powerful tool that can discover sensitive endpoints. Use responsibly:

- Only test systems you own or have explicit permission to test
- Respect rate limits and avoid overwhelming target systems
- Be mindful of the data you collect and how you store it
- Follow responsible disclosure practices for any vulnerabilities found

Happy enumerating! 🔍