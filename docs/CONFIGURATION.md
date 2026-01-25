# Enumeraite Configuration Guide

This document provides detailed information on configuring Enumeraite for optimal performance.

## Configuration Hierarchy

Enumeraite uses the following configuration precedence (highest to lowest):

1. Command-line arguments
2. Environment variables
3. Configuration files
4. Default values

## Configuration File Locations

Enumeraite searches for configuration files in the following order:

1. `./enumeraite.json` (current directory)
2. `~/.config/enumeraite/config.json` (user config directory)
3. `~/.enumeraite.json` (user home directory)

The first file found is used. If no configuration file is found, environment variables and defaults are used.

## Command-Line Arguments

For quick model selection without editing configuration files, use these CLI arguments:

```bash
# Select specific provider and model
enumeraite batch --target example.com --input paths.txt --provider claude --model anthropic/claude-sonnet-4-5

# Use different models with same provider
enumeraite batch --target example.com --input paths.txt --model anthropic/claude-opus-4-5
enumeraite batch --target example.com --input paths.txt --model gpt-4

# Works with continuous mode too
enumeraite continuous --target example.com --input paths.txt --model anthropic/claude-sonnet-4

# Show token usage and cost estimates (NEW!)
enumeraite batch --target example.com --input paths.txt --model anthropic/claude-sonnet-4 --debug
```

**Available Arguments:**
- `--provider`: Choose AI provider (`openai` or `claude`)
- `--model`: Override the model (e.g., `anthropic/claude-sonnet-4`, `anthropic/claude-sonnet-4-5`, `gpt-4`)
- `--debug`: Show token usage, cost estimates, and detailed debug information

**Popular Models:**
- **Claude**: `anthropic/claude-sonnet-4` (default), `anthropic/claude-sonnet-4-5`, `anthropic/claude-opus-4-5`
- **OpenAI**: `gpt-4` (default), `gpt-4-turbo`, `gpt-3.5-turbo`

## Full Configuration Schema

```json
{
  "providers": {
    "openai": {
      "api_key": "sk-...",
      "model": "gpt-4",
      "max_tokens": 1000,
      "temperature": 0.7,
      "organization": "org-..."
    },
    "claude": {
      "api_key": "sk-ant-...",
      "model": "anthropic/claude-sonnet-4",
      "max_tokens": 1000,
      "temperature": 0.7
    }
  },
  "default_provider": "openai",
  "http_validator": {
    "timeout": 10,
    "max_concurrent": 20,
    "user_agent": "Enumeraite/0.1.0",
    "follow_redirects": true,
    "verify_ssl": true,
    "retry_attempts": 3,
    "retry_delay": 1.0
  },
  "generation": {
    "min_confidence": 0.1,
    "max_duplicate_ratio": 0.8,
    "path_similarity_threshold": 0.9
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": null
  }
}
```

## Environment Variables

All configuration options can be set via environment variables using the pattern `ENUMERAITE_<SECTION>_<OPTION>`:

### Provider API Keys
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Default Provider
```bash
export ENUMERAITE_DEFAULT_PROVIDER="openai"
```

### HTTP Validator Settings
```bash
export ENUMERAITE_HTTP_TIMEOUT="10"
export ENUMERAITE_MAX_CONCURRENT="20"
export ENUMERAITE_USER_AGENT="Enumeraite/0.1.0"
export ENUMERAITE_FOLLOW_REDIRECTS="true"
export ENUMERAITE_VERIFY_SSL="true"
export ENUMERAITE_RETRY_ATTEMPTS="3"
export ENUMERAITE_RETRY_DELAY="1.0"
```

### Generation Settings
```bash
export ENUMERAITE_MIN_CONFIDENCE="0.1"
export ENUMERAITE_MAX_DUPLICATE_RATIO="0.8"
export ENUMERAITE_PATH_SIMILARITY_THRESHOLD="0.9"
```

### Logging Settings
```bash
export ENUMERAITE_LOG_LEVEL="INFO"
export ENUMERAITE_LOG_FILE="/path/to/log/file.log"
```

## Provider-Specific Configuration

### OpenAI Provider

```json
{
  "providers": {
    "openai": {
      "api_key": "sk-...",
      "model": "gpt-4",
      "max_tokens": 1000,
      "temperature": 0.7,
      "organization": "org-...",
      "base_url": "https://api.openai.com/v1",
      "timeout": 30
    }
  }
}
```

**Available Models:**
- `gpt-4` (recommended for best quality)
- `gpt-4-turbo`
- `gpt-3.5-turbo` (faster, lower cost)

**Parameters:**
- `api_key`: Your OpenAI API key
- `model`: Model to use for generation
- `max_tokens`: Maximum tokens in response
- `temperature`: Creativity level (0.0-1.0)
- `organization`: Optional organization ID
- `base_url`: API base URL (for custom endpoints)
- `timeout`: Request timeout in seconds

### Claude Provider

```json
{
  "providers": {
    "claude": {
      "api_key": "sk-ant-...",
      "model": "anthropic/claude-sonnet-4",
      "max_tokens": 1000,
      "temperature": 0.7,
      "timeout": 30
    }
  }
}
```

**Available Models (2026):**
- `anthropic/claude-sonnet-4-5` (newest flagship model, excellent coding)
- `anthropic/claude-sonnet-4` (recommended default, balanced performance)
- `anthropic/claude-opus-4-5` (highest quality for complex tasks)
- `anthropic/claude-haiku-3-5` (fastest, cost-effective)

**Parameters:**
- `api_key`: Your Anthropic API key
- `model`: Claude model to use
- `max_tokens`: Maximum tokens in response
- `temperature`: Creativity level (0.0-1.0)
- `timeout`: Request timeout in seconds

## HTTP Validator Configuration

```json
{
  "http_validator": {
    "timeout": 10,
    "max_concurrent": 20,
    "user_agent": "Enumeraite/0.1.0",
    "follow_redirects": true,
    "verify_ssl": true,
    "retry_attempts": 3,
    "retry_delay": 1.0,
    "headers": {
      "Accept": "application/json",
      "X-Custom-Header": "value"
    }
  }
}
```

**Parameters:**
- `timeout`: Request timeout in seconds
- `max_concurrent`: Maximum concurrent requests
- `user_agent`: HTTP User-Agent header
- `follow_redirects`: Whether to follow HTTP redirects
- `verify_ssl`: Whether to verify SSL certificates
- `retry_attempts`: Number of retry attempts for failed requests
- `retry_delay`: Delay between retries in seconds
- `headers`: Additional HTTP headers to send

## Generation Configuration

```json
{
  "generation": {
    "min_confidence": 0.1,
    "max_duplicate_ratio": 0.8,
    "path_similarity_threshold": 0.9,
    "enable_learning": true,
    "context_window": 50
  }
}
```

**Parameters:**
- `min_confidence`: Minimum confidence score for generated paths
- `max_duplicate_ratio`: Maximum ratio of duplicates before filtering
- `path_similarity_threshold`: Similarity threshold for duplicate detection
- `enable_learning`: Enable adaptive learning from discovered paths
- `context_window`: Number of recent paths to include in context

## Example Configurations

### Minimal Configuration
```json
{
  "providers": {
    "openai": {
      "api_key": "sk-..."
    }
  }
}
```

### Development Configuration
```json
{
  "providers": {
    "openai": {
      "api_key": "sk-...",
      "model": "gpt-3.5-turbo",
      "temperature": 0.8
    }
  },
  "http_validator": {
    "timeout": 5,
    "max_concurrent": 10,
    "verify_ssl": false
  },
  "logging": {
    "level": "DEBUG"
  }
}
```

### Production Configuration
```json
{
  "providers": {
    "openai": {
      "api_key": "sk-...",
      "model": "gpt-4",
      "max_tokens": 1000,
      "temperature": 0.7
    },
    "claude": {
      "api_key": "sk-ant-...",
      "model": "anthropic/claude-sonnet-4",
      "max_tokens": 1000,
      "temperature": 0.7
    }
  },
  "default_provider": "openai",
  "http_validator": {
    "timeout": 10,
    "max_concurrent": 50,
    "user_agent": "Enumeraite/0.1.0",
    "retry_attempts": 3
  },
  "generation": {
    "min_confidence": 0.3,
    "enable_learning": true
  },
  "logging": {
    "level": "INFO",
    "file": "/var/log/enumeraite.log"
  }
}
```

## Security Best Practices

### API Key Management
1. **Never commit API keys to version control**
2. Use environment variables in production
3. Rotate API keys regularly
4. Use least-privilege API key permissions

### Configuration Security
1. Set appropriate file permissions (600) on config files
2. Use secure directories for configuration
3. Encrypt sensitive configuration data
4. Audit configuration changes

### Network Security
1. Use HTTPS for all API endpoints
2. Verify SSL certificates in production
3. Use appropriate User-Agent strings
4. Respect rate limits and robots.txt

## Troubleshooting

### Common Issues

**API Key Not Found:**
```
Configuration error: No valid providers configured
```
Solution: Set API keys in config file or environment variables.

**Permission Denied:**
```
Configuration error: Permission denied reading config file
```
Solution: Check file permissions and ownership.

**Invalid JSON:**
```
Configuration error: Invalid JSON in config file
```
Solution: Validate JSON syntax using a JSON validator.

**Rate Limit Exceeded:**
```
Provider error: Rate limit exceeded
```
Solution: Reduce concurrent requests or add delays.

### Debug Mode
Enable debug logging to troubleshoot configuration issues:

```bash
export ENUMERAITE_LOG_LEVEL="DEBUG"
enumeraite batch --target example.com --input paths.txt
```

This will show detailed information about configuration loading and processing.