<p align="center">
  <img src="assets/enumeraite_logo.png" alt="Enumeraite Logo" width="400"/>
</p>

<h1 align="center">enumer<span style="color: #ef4444;">ai</span>te</h1>

<p align="center">
  <strong>AI-Powered Web Attack Surface Enumeration</strong>
</p>

<p align="center">
  <em>Proof-of-concept research demonstrating the future of intelligent enumeration.</em><br/>
  <strong>Traditional wordlists are dead. AI-driven discovery is the future.</strong>
</p>

<p align="center">
  <a href="https://github.com/oz9un/enumeraite/actions"><img src="https://img.shields.io/badge/tests-62%20passed-brightgreen?style=flat-square" alt="Tests"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/python-3.9+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License"></a>
  <a href="https://huggingface.co/enumeraite"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-research%20models-yellow?style=flat-square" alt="HuggingFace"></a>
</p>

<p align="center">
  <a href="https://enumeraite.com">🌐 Website</a> •
  <a href="https://huggingface.co/enumeraite">🤗 Models</a> •
  <a href="https://github.com/oz9un/enumeraite">📦 GitHub</a> •
  <a href="https://www.youtube.com/watch?v=IzsBS_E2RVY">📹 Talk</a>
</p>

---

<p align="center">
  <strong>Research presented at DEFCON 33 Recon Village</strong><br/>
  by <a href="https://github.com/oz9un">Özgün Kültekin</a> (<a href="https://x.com/oz9un">@oz9un</a>)
</p>

---

## 📺 Watch the Talk

<p align="center">
  <a href="https://www.youtube.com/watch?v=IzsBS_E2RVY">
    <img src="https://img.youtube.com/vi/IzsBS_E2RVY/maxresdefault.jpg" alt="Enumeraite Talk" width="600"/>
  </a>
</p>

<p align="center">
  <em>Click to watch the presentation</em>
</p>

---

## Features

### Two Modes of Operation

| Mode | Command | Purpose |
|------|---------|---------|
| **Generate** | `enumeraite generate` | Bulk generation from wordlists - feed it known paths/subdomains, get intelligent variants |
| **Analyze** | `enumeraite analyze` | Deep analysis of single targets - understand patterns and generate context-aware results |

### Capabilities

| Feature | Description |
|---------|-------------|
| **Path Generation** | Generate API endpoints from known paths using AI pattern recognition |
| **Subdomain Generation** | Generate subdomains based on naming patterns and conventions |
| **DNS Validation** | Validate generated subdomains via DNS resolution (`--validate`) |
| **HTTP Validation** | Check HTTP/HTTPS response for validated subdomains (`--check-http`) |
| **Pattern Analysis** | Deep decomposition of complex naming patterns (analyze mode) |
| **Function-based Discovery** | Find endpoints for specific functionality like "user deletion" or "admin ops" |
| **Debug Mode** | Track token usage and cost estimation (`--debug`) |
| **Tool Integration** | Pipe output directly to ffuf, gobuster, dirb, nuclei |

### Supported Providers

| Provider | Setup | Best For |
|----------|-------|----------|
| **Claude** | `export ANTHROPIC_API_KEY='...'` | Production use (recommended) |
| **OpenAI** | `export OPENAI_API_KEY='...'` | Production use |
| **HuggingFace** | No setup needed | Free experimentation (limited quality) |

## Quick Start

### 30-Second Start

```bash
# Install enumeraite
pip install "enumeraite[claude]"

# Set API key
export ANTHROPIC_API_KEY='your-key-here'

# Create sample paths file
echo -e "/api/users\n/api/auth/login\n/admin/dashboard" > paths.txt

# Generate intelligent variants
enumeraite generate path --input paths.txt --count 20
```

### Installation

**Option 1: PyPI (Recommended)**
```bash
pip install enumeraite
```

**Option 2: From Source (Development)**
```bash
git clone https://github.com/oz9un/enumeraite.git
cd enumeraite
pip install -e .
```

**Optional Provider Dependencies:**
```bash
# For Claude support
pip install "enumeraite[claude]"

# For OpenAI support
pip install "enumeraite[openai]"

# For HuggingFace support
pip install "enumeraite[huggingface]"

# Install all providers
pip install "enumeraite[all]"
```

### Setup

**Option 1: Claude (Recommended)**
```bash
# Get API key from https://console.anthropic.com/
export ANTHROPIC_API_KEY='your-api-key-here'

# Generate paths with Claude
enumeraite generate path --input paths.txt --provider claude --count 20
```

**Option 2: OpenAI**
```bash
# Get API key from https://platform.openai.com/api-keys
export OPENAI_API_KEY='your-api-key-here'

# Generate paths with GPT-4
enumeraite generate path --input paths.txt --provider openai --count 20
```

**Option 3: Local Models (Experimental)**
```bash
# No API key needed, but quality is limited
enumeraite generate path --input paths.txt --provider huggingface --model enumeraite/Enumeraite-x-Qwen3-4B-Path --count 20
```

### Basic Usage Examples

**Path Discovery:**
```bash
# Start with known endpoints
echo "/api/users
/api/auth/login
/admin/dashboard" > known_paths.txt

# Generate intelligent variants
enumeraite generate path --input known_paths.txt --provider claude --count 25
```

**Subdomain Discovery:**
```bash
# Known subdomains
echo "api.example.com
admin.example.com
staging.example.com" > known_subs.txt

# Generate with DNS validation
enumeraite generate subdomain --input known_subs.txt --provider claude --validate --count 30
```

### Understanding Token Usage and Models

```bash
# Monitor token usage with debug flag
enumeraite generate path --input paths.txt --provider claude --count 25 --debug

# Use specific models
enumeraite generate path --input paths.txt --provider openai --model gpt-4 --count 20
enumeraite generate path --input paths.txt --provider claude --model anthropic/claude-sonnet-4 --count 20
enumeraite generate path --input paths.txt --provider huggingface --model enumeraite/Enumeraite-x-Qwen3-4B-Subdomain --count 15
```

## Examples

### Bulk Path Generation

```bash
enumeraite generate path --input my_paths.txt --provider claude --count 20
```

<p align="center">
  <img src="assets/demo_images/bulk_path.png" alt="Bulk Path Generation Example" width="700"/>
</p>

### Subdomain Generation with DNS Validation

```bash
enumeraite generate subdomain --input my_subdomains.txt --provider claude --validate --count 30
```

<p align="center">
  <img src="assets/demo_images/subdomain_bulk.png" alt="Subdomain Generation Example" width="700"/>
</p>

### Path Function Analysis

```bash
enumeraite analyze path "/api/Usr_crt" --function "user deletion" --provider claude
```

<p align="center">
  <img src="assets/demo_images/analyze_path.png" alt="Path Function Analysis Example" width="700"/>
</p>

### Subdomain Pattern Analysis

```bash
enumeraite analyze subdomain "activateiphone-use1-cx02.example.com" --provider claude
```

<p align="center">
  <img src="assets/demo_images/analyze_subdomain.png" alt="Subdomain Pattern Analysis Example" width="700"/>
</p>

## Tool Integration

Enumeraite output is designed to work seamlessly with popular fuzzing tools.

### Pipe to ffuf

```bash
# Generate paths and fuzz directly
enumeraite generate path -i known_paths.txt -c 100 | ffuf -w - -u https://target.com/FUZZ

# Save to file first, then use
enumeraite generate path -i known_paths.txt -o wordlist.txt
ffuf -w wordlist.txt -u https://target.com/FUZZ
```

### Pipe to gobuster

```bash
enumeraite generate path -i known_paths.txt | gobuster dir -u https://target.com -w -
```

### With nuclei

```bash
# Generate subdomains, validate, then scan
enumeraite generate subdomain -i subs.txt --validate -o live_subs.txt
nuclei -l live_subs.txt -t cves/
```

## Command Reference

### generate path
```
enumeraite generate path -i <input> [options]

Options:
  -i, --input PATH     Input file with known paths (required)
  -o, --output PATH    Output file (default: stdout)
  -c, --count INT      Number to generate (default: 50)
  --provider TEXT      AI provider: claude, openai, huggingface
  --model TEXT         Specific model to use
  --debug              Show token usage and cost
```

### generate subdomain
```
enumeraite generate subdomain -i <input> [options]

Options:
  -i, --input PATH     Input file with known subdomains (required)
  -o, --output PATH    Output file (default: stdout)
  -c, --count INT      Number to generate (default: 50)
  --provider TEXT      AI provider: claude, openai, huggingface
  --model TEXT         Specific model to use
  --validate           Enable DNS validation
  --check-http         Check HTTP response (requires --validate)
  --debug              Show token usage and cost
```

### analyze path
```
enumeraite analyze path <path> -f <function> [options]

Options:
  -f, --function TEXT  Functionality to find (required)
  -c, --count INT      Number of variants (default: 20)
  -o, --output PATH    Output file (default: stdout)
  --provider TEXT      AI provider: claude, openai, huggingface
  --debug              Show debug info
```

### analyze subdomain
```
enumeraite analyze subdomain <subdomain> [options]

Options:
  -c, --count INT      Number of variants (default: 20)
  -o, --output PATH    Output file (default: stdout)
  --provider TEXT      AI provider: claude, openai, huggingface
  --debug              Show debug info
```

## Future Vision

This research opens several exciting directions:

### Near-term Improvements:
- **Better fine-tuned models** trained on real application data
- **Target-specific wordlist generation** based on technology stack
- **Response-aware fuzzing** that adapts based on HTTP responses
- **Integration with existing tools** (ffuf, dirb, gobuster)

### Long-term Potential:
- **RAG-enhanced models** with application-specific knowledge bases
- **Multi-modal analysis** incorporating HTML, JavaScript, and API schemas

### Research Applications:
- Academic study of AI in offensive security
- Benchmark for evaluating enumeration approaches
- Foundation for specialized security AI models

## Quality Comparison

| Model | Quality | Consistency | Cost | Use Case |
|-------|---------|-------------|------|----------|
| **Claude Sonnet** | ⭐⭐⭐⭐⭐ | Excellent | Low | Production research |
| **GPT-4** | ⭐⭐⭐⭐⭐ | Very Good | Medium | Production research |
| **Custom Enumeraite Models** | ⭐⭐⚫⚫⚫ | Poor | Free | Demo/testing only |

## Contributing to Research

We welcome contributions that advance the methodology:

- **Model improvements** and training data
- **Integration with existing tools**
- **Novel enumeration techniques**
- **Evaluation metrics and benchmarks**
- **Real-world case studies**

## License

This research project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>⚠️ Ethical Use Disclaimer</strong><br/>
  This research tool is intended for authorized security testing and academic research only.<br/>
  Users are responsible for ensuring they have proper permission to test target systems.<br/>
  <br/>
  <strong>Research Status</strong><br/>
  This is proof-of-concept research software. Results may vary.<br/>
  For production security testing, combine with traditional methods.
</p>

---

<p align="center">
  <em>"The future of enumeration is not about having the biggest wordlist,<br/>
  but about having the smartest approach."</em>
</p>

---

<p align="center">
  For detailed documentation, examples, and advanced usage:<br/>
  <strong><a href="https://enumeraite.com">Visit enumeraite.com</a></strong>
</p>
