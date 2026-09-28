# 🎙️ AI Voice Receptionist & Customer Support Agent

An intelligent, real-time voice-first customer assistant built for inbound customer support, product inquiries, business knowledge retrieval (RAG), and smart escalation. 

Powered by **LiveKit Agents**, **Groq (Llama / OSS models)**, **Deepgram Nova-3**, **Inworld TTS**, **Docling**, and **Qdrant Vector Database**.

---

## 🚀 Key Features

- **⚡ Real-Time Audio Streaming**: Sub-second end-to-end voice latency with WebRTC via LiveKit Agents.
- **🧠 Fast LLM Reasoning**: Accelerated responses using Groq's high-speed inference engine.
- **📚 Multimodal RAG (Retrieval-Augmented Generation)**:
  - Rich document parsing (PDF, DOCX, TXT, MD, HTML) using **IBM Docling**.
  - Dense vector search with **Sentence Transformers** (`all-MiniLM-L6-v2`) and **Qdrant**.
- **🎯 STT Keyterm Boosting**: Deepgram Nova-3 STT tuned with brand/product keywords for high recognition accuracy on specialized terminology.
- **🗣️ Phonetic Pronunciation Mapping**: Custom phonetic dictionary to ensure natural, correct speech synthesis from Inworld TTS.
- **📋 Inquiry & Lead Capture**: Automatically logs caller inquiries, contact information, and callback requests.
- **🚨 Smart Human Escalation**: Gracefully detects when a caller requires human assistance and initiates a handoff workflow.
- **📝 Call Transcript Recording**: Automatically captures and exports structured transcripts of caller sessions.
- **⚙️ Fully Configurable**: Adaptable to any business, persona, or brand via `agent_config.yaml`.

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Voice Pipeline** | [LiveKit Agents](https://livekit.io/) | Real-time WebRTC audio transport & session orchestration |
| **LLM Engine** | [Groq](https://groq.com/) | Low-latency inference (`openai/gpt-oss-120b` or Llama models) |
| **Orchestration** | [LangChain](https://www.langchain.com/) | Agent graph, tool routing, and memory handling |
| **STT (Speech-to-Text)** | [Deepgram](https://deepgram.com/) (Nova-3) | Real-time speech transcription with keyterm boosting |
| **TTS (Text-to-Speech)** | [Inworld AI](https://inworld.ai/) (TTS-2) | Expressive voice synthesis with phonetic mapping |
| **Document Ingestion** | [Docling](https://github.com/DS4SD/docling) | Hierarchical layout-aware document chunking |
| **Vector Database** | [Qdrant](https://qdrant.tech/) | Local/cloud dense vector index for knowledge base RAG |
| **Embeddings** | [Sentence Transformers](https://sbert.net/) | `all-MiniLM-L6-v2` dense embedding generation |

---

## 📂 Project Structure

```text
├── agent_config.yaml       # Business profile, persona guidelines, and pronunciation rules
├── config.py               # Pydantic environment and runtime settings
├── db.py                   # Qdrant client & SentenceTransformer singletons
├── ingest.py               # Document converter & vector DB embedding ingestion script
├── pyproject.toml          # Project dependencies & build configuration
├── .env.example            # Template for environment secrets
├── docs/                   # Knowledge base documents (PDFs, docs, markdown)
├── data/                   # Local Qdrant vector storage and call transcripts
├── langchain_agent/        # LangChain graph, tools (search, capture, escalate)
└── voice_agent/            # LiveKit entrypoint, audio session handler, transcript recorder
```

---

## 📋 Prerequisites

- **Python**: `>= 3.12`
- **[LiveKit Cloud](https://livekit.io/) Account**: URL, API Key, and API Secret
- **[Groq Cloud](https://console.groq.com/) Account**: API Key
- *(Optional)* Deepgram / Inworld API credentials if not proxied through LiveKit inference plugins.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO_NAME>.git
cd <YOUR_REPO_NAME>
```

### 2. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Using `pip`:
```bash
pip install -e .
```
*Or with `uv` (recommended for faster resolution):*
```bash
uv sync
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**macOS / Linux:**
```bash
cp .env.example .env
```

Open `.env` and fill in your API credentials:
```env
LIVEKIT_URL=wss://your-project-id.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

GROQ_API_KEY=your_groq_api_key
```

---

## 📖 Knowledge Base Ingestion (RAG)

1. Place your company brochures, FAQs, product sheets, or documentation into the `docs/` folder (supported: `.pdf`, `.docx`, `.txt`, `.md`, `.html`).
2. Run the ingestion pipeline:

```bash
python ingest.py
```

This will:
- Parse documents using **Docling**.
- Chunk the content while preserving header hierarchies.
- Generate vector embeddings using `all-MiniLM-L6-v2`.
- Index chunks into local **Qdrant** database under `./data/qdrant`.

---

## 🎧 Running the Voice Agent

### Option 1: Local Console Mode (Interactive Microphone Testing)
Test the agent directly in your terminal using your computer's microphone and speakers:

```bash
python -m voice_agent.main console
```

### Option 2: LiveKit Cloud / WebRTC Mode
Start the agent worker to connect to your LiveKit room:

```bash
# Development mode with auto-reload:
python -m voice_agent.main dev

# Production worker:
python -m voice_agent.main start
```

---

## 🔧 Customizing for Your Business

Edit `agent_config.yaml` to customize the agent for any company or client:

```yaml
company_name: "Your Company"
agent_name: "Alex"
welcome_greeting: "Greet the caller warmly as {agent_name} from {company_name} and ask how you can help."

persona:
  role: "You are {agent_name}, a friendly customer service assistant representing {company_name}."
  tone: "Warm, natural, and concise. Keep responses to 1-2 spoken sentences."
  guidelines:
    - "Use the search_knowledge_base tool to look up accurate information."
    - "Use capture_inquiry to collect caller details when requested."
    - "Use escalate_to_human when a caller requests human support."

# Boost Deepgram accuracy for custom terms/names
stt_keyterms:
  - "YourBrandName"
  - "SpecificProduct"

# Phonetic corrections for TTS engine
pronunciation_map:
  "YourBrandName": "Your-Brand-Naym"
```

---

## 📝 Example Voice Interactions

| Caller Request | Agent Behavior |
|---|---|
| *"Can you tell me about your pricing plans?"* | Queries the indexed documents via vector search (`search_knowledge_base`) and summarizes the answer concisely. |
| *"I'd like to schedule a callback regarding enterprise service."* | Collects the caller's contact details and logs an inquiry (`capture_inquiry`). |
| *"I need to speak with a human supervisor."* | Initiates the escalation workflow and logs a support ticket (`escalate_to_human`). |
