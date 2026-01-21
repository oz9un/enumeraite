# 🔍 Enumeraite Seed-Based Path Generator - MVP Design

> **Project**: AI-Assisted Web Attack Surface Enumeration
> **Component**: Seed-Based Path Generator (First Tool)
> **Date**: 2026-01-22

## 🎯 Overview

The seed-based path generator expands API attack surfaces from known endpoints using AI pattern recognition, contextual understanding, and continuous learning. Users provide known paths, and the tool generates new potential endpoints for testing.

## 🏗️ Architecture

### Core Modes

1. **Batch Mode** - Generate paths once, output results
2. **Continuous Mode** - Intelligent discovery loop with live updates

### Provider Plugin System

```python
class BaseProvider:
    def __init__(self, config): pass
    def generate_paths(self, known_paths, target, count=50): pass
    def get_confidence_scores(self, paths, known_paths): pass
```

**Initial Providers:**
- OpenAIProvider (comprehensive single prompt)
- ClaudeProvider (comprehensive single prompt)
- EnumeraiteProvider (Phase 2 - custom model handling)

### Intelligence Stack

- **Pattern Analysis**: REST conventions, naming schemes
- **Contextual Understanding**: Domain concepts, role-based endpoints
- **Confidence Scoring**: Multi-factor blend (model probability + pattern strength)
- **Continuous Learning**: Real-time adaptation from valid discoveries

## 💻 Command Interface

```bash
# Batch mode (MVP Phase 1)
enumeraite batch --target example.com --input paths.txt [--count 50] [--provider openai]

# Advanced output with metadata
enumeraite batch --target example.com --input paths.txt --advanced [--validate]

# Continuous discovery mode (MVP Phase 1)
enumeraite continuous --target example.com --input paths.txt --duration 30m --batch-size 20 --max-empty-rounds 5
```

## 📁 Input/Output Format

### Input
```
# paths.txt
/api/user/profile
/api/user/settings
/api/admin/dashboard
```

### Output - Simple (Default)
```
/api/user/preferences
/api/user/notifications
/api/admin/users
/api/admin/analytics
```

### Output - Advanced (--advanced)
```json
{
  "target": "example.com",
  "provider": "openai",
  "generated_count": 4,
  "generated_paths": [
    {
      "path": "/api/user/preferences",
      "confidence": 0.85,
      "method": "pattern",
      "verified": true,
      "status": 200
    },
    {
      "path": "/api/user/notifications",
      "confidence": 0.72,
      "method": "context",
      "verified": false
    }
  ]
}
```

## 🔄 Continuous Discovery Flow

1. **Initialize**: Load seed paths from input file
2. **Generate**: Create batch of new paths using current working set
3. **Validate**: Test paths with basic HTTP probing (optional)
4. **Update**: Append valid paths to input file in real-time
5. **Learn**: Use expanded set for next generation round
6. **Repeat**: Until duration expires OR max empty rounds reached

### Live User Experience
- Real-time console output showing discoveries
- Input file continuously updated (`tail -f paths.txt`)
- Progress indicators (round #, discoveries, success rate)

## ⚙️ Configuration

### Provider Selection Priority
1. User-specified via `--provider` flag
2. First available provider: OpenAI → Claude → Enumeraite
3. Config file for API keys and preferences

### Validation (Optional)
- Basic HTTP GET/POST probing
- Configurable HTTP methods
- Simple success heuristics (non-404 responses)

## 🎯 MVP Scope

**Phase 1 (Initial Release):**
- Batch mode with OpenAI/Claude providers
- Continuous discovery mode
- Basic HTTP validation
- Simple + Advanced output formats

**Phase 2 (Post-MVP):**
- Enumeraite custom model integration
- Sophisticated validation (auth, headers)
- Learning-based retraining
- Additional output formats

## 🔍 Example Use Case

```bash
# Start with basic recon findings
echo "/api/user/profile" > discovered.txt
echo "/api/admin/dashboard" >> discovered.txt

# Run continuous discovery for 30 minutes
enumeraite continuous --target example.com --input discovered.txt --duration 30m

# Watch live discoveries
tail -f discovered.txt

# Result: discovered.txt grows from 2 paths to potentially 50+ valid endpoints
```

## 🚀 Success Metrics

- **Effectiveness**: % of generated paths that return non-404 responses
- **Coverage**: New attack surface discovered vs existing tools
- **Usability**: Integration with existing recon workflows
- **Performance**: Speed of generation and validation

---

*This design represents the foundation for enumeraite's first AI-powered reconnaissance tool, focusing on practical attack surface expansion for security researchers and penetration testers.*