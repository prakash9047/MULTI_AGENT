# AI Tool Usage Documentation

This document logs the AI tools, models, and parameters used in the Finance Assistant project.

## Language Models

### Primary LLM
- Model: `mistralai/Mistral-7B-Instruct-v0.1`
- Framework: HuggingFace Transformers
- Parameters:
  - max_length: 512
  - temperature: 0.7
  - top_p: 0.9
  - repetition_penalty: 1.2

### Embeddings
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Framework: Sentence Transformers
- Parameters:
  - chunk_size: 1000
  - chunk_overlap: 200

## Voice Processing

### Speech-to-Text
- Model: Whisper
- Version: base
- Parameters:
  - sample_rate: 16000
  - language: en

### Text-to-Speech
- Model: Coqui TTS
- Version: tts_models/en/ljspeech/tacotron2-DDC
- Parameters:
  - sample_rate: 22050

## Vector Store

### FAISS
- Index Type: L2
- Parameters:
  - nlist: 100
  - nprobe: 10

## Agent Frameworks

### CrewAI
- Version: 0.1.0
- Configuration:
  - max_concurrent_agents: 5
  - request_timeout: 30
  - cache_ttl: 300

### LangChain
- Version: 0.0.335
- Components:
  - Chains
  - Agents
  - Memory
  - Prompts

## Data Processing

### Document Processing
- Chunking: RecursiveCharacterTextSplitter
- Parameters:
  - chunk_size: 1000
  - chunk_overlap: 200

### RAG Pipeline
- Retrieval:
  - top_k: 5
  - confidence_threshold: 0.7

## Performance Metrics

### Latency
- API Agent: <500ms
- Voice STT: <2s
- RAG Retrieval: <1s
- Full Pipeline: <5s

### Throughput
- API Requests: 100/min
- RAG Queries: 50/min
- Voice Processing: Real-time

## Error Handling

### Fallback Mechanisms
- LLM: Retry with lower temperature
- STT: Fallback to text input
- RAG: Confidence-based fallback
- API: Circuit breaker pattern

## Monitoring

### Metrics
- Response time
- Error rates
- Resource usage
- Agent performance

### Logging
- Request/response logs
- Error logs
- Performance metrics
- Agent interactions 