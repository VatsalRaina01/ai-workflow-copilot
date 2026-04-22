# AI Workflow Copilot — Deep Dive

A comprehensive technical explanation of every RAG strategy, chunking mechanism, search algorithm, agent architecture, and safety layer used in this project.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Document Ingestion Pipeline](#2-document-ingestion-pipeline)
   - 2.1 [Multi-Format Parsing](#21-multi-format-parsing)
   - 2.2 [Chunking Strategy — RecursiveCharacterTextSplitter](#22-chunking-strategy--recursivecharactertextsplitter)
   - 2.3 [Embedding Generation](#23-embedding-generation)
   - 2.4 [Dual-Index Storage (ChromaDB + BM25)](#24-dual-index-storage-chromadb--bm25)
3. [Retrieval / Search Strategies](#3-retrieval--search-strategies)
   - 3.1 [Semantic Search (Cosine Similarity)](#31-semantic-search-cosine-similarity)
   - 3.2 [BM25 Keyword Search](#32-bm25-keyword-search)
   - 3.3 [Hybrid Search with Reciprocal Rank Fusion (RRF)](#33-hybrid-search-with-reciprocal-rank-fusion-rrf)
4. [Multi-Agent Orchestration](#4-multi-agent-orchestration)
   - 4.1 [BaseAgent (Abstract Base Class)](#41-baseagent-abstract-base-class)
   - 4.2 [RAG Agent](#42-rag-agent)
   - 4.3 [Summarization Agent](#43-summarization-agent)
   - 4.4 [Code Agent](#44-code-agent)
   - 4.5 [Agent Router — LLM-Based Intent Detection](#45-agent-router--llm-based-intent-detection)
5. [Conversation Memory](#5-conversation-memory)
6. [Generation — Prompt Construction & LLM Calling](#6-generation--prompt-construction--llm-calling)
7. [Streaming (Server-Sent Events)](#7-streaming-server-sent-events)
8. [Guardrails — Security & Safety Layer](#8-guardrails--security--safety-layer)
   - 8.1 [Prompt Injection Detection](#81-prompt-injection-detection)
   - 8.2 [PII Redaction](#82-pii-redaction)
   - 8.3 [Rate Limiting](#83-rate-limiting)
9. [RAG Evaluation — LLM-as-Judge](#9-rag-evaluation--llm-as-judge)
10. [Workflow Engine — Multi-Step Prompt Chaining](#10-workflow-engine--multi-step-prompt-chaining)
11. [Configuration & Tunables](#11-configuration--tunables)
12. [End-to-End Data Flow (Putting It All Together)](#12-end-to-end-data-flow-putting-it-all-together)

---

## 1. High-Level Architecture

The project follows a **three-tier architecture**:

```
Frontend (React + Vite)  ──REST/SSE──►  Backend (FastAPI)  ──SDK──►  GitHub Models API (LLM + Embeddings)
                                              │
                                         ┌────┴────┐
                                     ChromaDB    BM25 Index
                                    (Persistent) (In-Memory)
```

**Backend layers** (inside `app/`):

| Layer | Directory | Responsibility |
|-------|-----------|----------------|
| **API** | `api/endpoints/` | FastAPI routes, request validation (Pydantic), SSE streaming |
| **Agents** | `agents/` | Multi-agent system — RAG, Summarize, Code agents + Router |
| **Services** | `services/` | Vector store, document parsing, guardrails, evaluation, LLM wrappers |
| **Core** | `core/` | Application configuration via Pydantic Settings |

Every service is instantiated as a **singleton** exported at module level, ensuring one shared instance across the application.

---

## 2. Document Ingestion Pipeline

This is the "write path" of the RAG system — how documents go from raw files to searchable vector/keyword indices.

### 2.1 Multi-Format Parsing

**File:** `services/document_parser.py` — Class `DocumentParser`

The parser supports four file formats via a unified `.parse(file_bytes, filename)` interface:

| Format | Parser | Details |
|--------|--------|---------|
| `.txt`, `.md`, `.text` | UTF-8 / Latin-1 decode | Raw text files decoded directly. Falls back to Latin-1 if UTF-8 fails. |
| `.pdf` | PyPDF2 `PdfReader` | Iterates all pages, extracts text per page. Adds `--- Page N ---` headers for context. Warns if no extractable text (scanned PDFs). |
| `.docx` | python-docx `Document` | Extracts paragraphs while **preserving heading hierarchy** (converts `Heading 1` → `# Title`, `Heading 2` → `## Title`, etc.). Also extracts **table content** as pipe-separated rows. |

**Why this matters for RAG:** The parser ensures structural information (headings, page boundaries, tables) is preserved in the extracted text. When the text is later chunked, these structural markers help the `RecursiveCharacterTextSplitter` make semantically meaningful splits.

### 2.2 Chunking Strategy — RecursiveCharacterTextSplitter

**File:** `services/vector_store.py` — configured inside `VectorStoreService.__init__()`

```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # Max characters per chunk
    chunk_overlap=50,      # Characters of overlap between consecutive chunks
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

#### How RecursiveCharacterTextSplitter Works

This is **LangChain's most popular text splitter** and is used here for its ability to create **semantically coherent chunks** rather than arbitrary fixed-size cuts. It works recursively through a hierarchy of separators:

1. **First attempt**: Split on `"\n\n"` (double newline = paragraph boundary)
   - This keeps full paragraphs together, which preserves semantic coherence.
   - Each resulting chunk is checked: if it's ≤ 500 characters, it's accepted.

2. **If a chunk is still too long**: Split on `"\n"` (single newline = line break)
   - Breaks long paragraphs into individual lines.

3. **If still too long**: Split on `". "` (period-space = sentence boundary)
   - Breaks long lines into individual sentences.

4. **If still too long**: Split on `" "` (space = word boundary)
   - Breaks long sentences into individual words.

5. **Last resort**: Split on `""` (character-level split)
   - Only used for extremely long words or tokens.

#### Why This Hierarchy?

The key insight is that the splitter **prefers to split at higher semantic boundaries first**. A paragraph split is better than a mid-sentence split. The result is chunks that are self-contained "thoughts" rather than abruptly truncated text.

#### Chunk Overlap

```
chunk_overlap=50
```

The 50-character overlap means that the last 50 characters of Chunk N are also included as the first 50 characters of Chunk N+1. This is critical because:

- It prevents information loss at chunk boundaries.
- If a key fact spans two chunks, the overlap ensures it appears in full in at least one.
- It improves retrieval recall — a query matching the boundary region will hit both chunks.

#### Configuration

| Parameter | Value | Effect |
|-----------|-------|--------|
| `CHUNK_SIZE` | 500 | Smaller chunks = more precise retrieval but less context. 500 is a good balance. |
| `CHUNK_OVERLAP` | 50 | 10% overlap ratio. Enough to bridge sentence boundaries. |

### 2.3 Embedding Generation

**File:** `services/github_models.py` — Class `GitHubModelsService`

After chunking, each text chunk is converted to a dense vector embedding:

```python
embeddings = github_models.get_embeddings(chunks)  # Batch embedding
```

- **Model:** `text-embedding-3-small` (OpenAI's latest small embedding model via GitHub Models API)
- **Method:** The OpenAI SDK `client.embeddings.create()` is called with all chunks in a single batch call for efficiency.
- **Retry Logic:** Exponential backoff (1s, 2s, 4s) for up to 3 attempts on API failure.
- **Output:** Each chunk becomes a 1536-dimensional float vector that captures its semantic meaning.

### 2.4 Dual-Index Storage (ChromaDB + BM25)

The system stores every chunk in **two separate indices** — this is the foundation for hybrid search:

#### ChromaDB (Semantic / Vector Index)

```python
self.client = chromadb.PersistentClient(path="./chroma_data")
self.collection = self.client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}  # Cosine similarity metric
)
```

- **Storage:** `PersistentClient` stores data on disk (`./chroma_data/`), so documents survive server restarts. Falls back to ephemeral in-memory mode if disk fails.
- **Indexing Algorithm:** HNSW (Hierarchical Navigable Small World) — an approximate-nearest-neighbor algorithm optimized for high-dimensional vectors.
- **Distance Metric:** Cosine similarity — measures the angle between two vectors regardless of magnitude.
- **What's Stored:** For each chunk: `(chunk_id, embedding_vector, raw_text, metadata)`.
- **Metadata per chunk:** `doc_id` (parent document UUID), `chunk_index` (position within document), `filename`, `file_type`.

#### BM25 Index (Keyword / Lexical Index)

```python
self.bm25_index = BM25Index(k1=1.5, b=0.75)
```

A custom in-memory BM25 index built from scratch (class `BM25Index`). Every chunk added to ChromaDB is also added here.

**On startup**, if ChromaDB already has documents (from a previous session), the BM25 index is rebuilt from ChromaDB data via `_rebuild_bm25_index()`.

---

## 3. Retrieval / Search Strategies

This is the "read path" — how user queries are matched to relevant chunks.

### 3.1 Semantic Search (Cosine Similarity)

**File:** `services/vector_store.py` — method `VectorStoreService.search()`

```python
query_embedding = github_models.get_embedding(query)
results = self.collection.query(
    query_embeddings=[query_embedding],
    n_results=top_k,
    include=["documents", "metadatas", "distances"]
)
```

**How it works:**
1. The user's query is embedded into a vector using the same `text-embedding-3-small` model.
2. ChromaDB performs approximate nearest neighbor (ANN) search using the HNSW index.
3. Results are ranked by cosine distance (converted to similarity: `score = 1 - distance`).

**Strengths:** Captures **semantic meaning** — the query "What are the benefits?" matches chunks containing "advantages" or "pros" even if those exact words aren't present.

**Weakness:** Misses exact keyword matches — the query "error code 404" might not match a chunk containing "HTTP 404 Not Found" if the embeddings aren't close enough.

### 3.2 BM25 Keyword Search

**File:** `services/vector_store.py` — class `BM25Index`

BM25 (Best Matching 25) is a probabilistic information retrieval algorithm based on **term frequency** and **inverse document frequency**.

#### The BM25 Scoring Formula

For each query token `q` in a document `d`:

```
score(q, d) = IDF(q) × [ tf(q,d) × (k1 + 1) ] / [ tf(q,d) + k1 × (1 - b + b × |d| / avgdl) ]
```

Where:
- **`tf(q,d)`** = frequency of query term `q` in document `d`
- **`IDF(q)`** = `log((N - df(q) + 0.5) / (df(q) + 0.5) + 1)` — inverse document frequency; rare terms score higher
- **`k1 = 1.5`** = term frequency saturation parameter (controls how quickly additional term occurrences diminish in value)
- **`b = 0.75`** = length normalization parameter (0 = no normalization, 1 = full normalization)
- **`|d|`** = document length (in tokens)
- **`avgdl`** = average document length across all documents
- **`N`** = total number of documents
- **`df(q)`** = number of documents containing term `q`

#### Tokenization

```python
def _tokenize(self, text: str) -> List[str]:
    return re.findall(r'\w+', text.lower())
```

Simple whitespace + lowercasing tokenization:
- Splits on non-word characters (extracts only alphanumeric tokens)
- Case-insensitive (everything lowercased)
- No stemming, lemmatization, or stop-word removal (kept simple for performance)

**Strengths:** Excels at **exact keyword matching** — perfect for queries containing specific terms, acronyms, error codes, or proper nouns.

**Weakness:** No semantic understanding — "automobile" won't match "car".

### 3.3 Hybrid Search with Reciprocal Rank Fusion (RRF)

**File:** `services/vector_store.py` — method `VectorStoreService.hybrid_search()`

This is the **core retrieval strategy** used by the RAG agent. It combines semantic and BM25 results using **Reciprocal Rank Fusion (RRF)**.

#### Step-by-Step Process

```python
def hybrid_search(self, query, top_k=5, semantic_weight=0.7):
```

**Step 1 — Fetch from both indices:**
```python
semantic_results = self.search(query, top_k=top_k * 2)   # Double the results
bm25_results = self.bm25_index.search(query, top_k=top_k * 2)
```
Both searches return `2× top_k` results (= 10 by default) to ensure sufficient candidates for fusion.

**Step 2 — Reciprocal Rank Fusion (RRF):**

RRF is a rank aggregation algorithm that combines ranked lists without needing to normalize raw scores across different scoring systems.

For each document, its RRF score is:

```
RRF_score = Σ [ weight / (k + rank + 1) ]
```

Where:
- `k = 60` (constant that dampens the influence of high-ranked items, standard in RRF literature)
- `rank` = 0-indexed position in the result list
- `weight` = `semantic_weight` (0.7) for semantic results, `1 - semantic_weight` (0.3) for BM25 results

**Example calculation** for a chunk that appears at rank 0 in semantic results and rank 2 in BM25:

```
RRF = 0.7/(60+0+1) + 0.3/(60+2+1) = 0.7/61 + 0.3/63 = 0.01148 + 0.00476 = 0.01624
```

**Step 3 — Sort and return top K:**
All unique chunks are sorted by their combined RRF score in descending order, and the top `top_k` (default 5) are returned.

#### Why Hybrid Search?

| Scenario | Semantic Only | BM25 Only | Hybrid |
|----------|:---:|:---:|:---:|
| "What are the benefits of this approach?" | ✅ | ❌ | ✅ |
| "error code ERR_NOT_FOUND_404" | ❌ | ✅ | ✅ |
| "How does authentication work?" (keyword + concept) | ✅ | ✅ | ✅✅ (boosted) |

Hybrid search gives you the **best of both worlds** — semantic understanding for paraphrased/conceptual queries AND exact matching for keyword/code queries.

#### Weight Configuration

```python
BM25_WEIGHT = 0.3  # ⟹ semantic_weight = 1 - 0.3 = 0.7
```

The 70/30 semantic-to-BM25 weight ratio reflects that:
- Most user questions are natural language (semantic excels)
- BM25 compensates for edge cases where exact keywords matter
- This ratio is configurable via the `BM25_WEIGHT` environment variable

---

## 4. Multi-Agent Orchestration

The project implements a **multi-agent architecture** where different specialized agents handle different types of queries.

### 4.1 BaseAgent (Abstract Base Class)

**File:** `agents/base.py`

```python
class BaseAgent(ABC):
    def __init__(self, name: str, description: str, agent_type: str)
    
    @abstractmethod
    async def process(self, query, context) -> Dict
    
    @abstractmethod
    def process_stream(self, query, context) -> Generator
    
    def get_system_prompt(self) -> str
    def get_info(self) -> Dict
```

This enforces a **polymorphic interface** — any agent can be swapped in/out without changing the router. Key OOP principles:
- **Abstraction:** The `process()` and `process_stream()` methods define the contract.
- **Polymorphism:** Each subclass implements these methods differently.
- **Encapsulation:** Agent-specific logic (prompts, retrieval, etc.) is self-contained.

### 4.2 RAG Agent

**File:** `agents/rag_agent.py`

The core document Q&A agent. Its `process()` flow:

```
Query → Hybrid Search → Build Context String → Construct Prompt → LLM Call → Response + Sources
```

1. **Retrieve:** Calls `vector_store.hybrid_search(query)` to get top-5 relevant chunks.
2. **Build Context:** Formats chunks into a labeled reference string:
   ```
   [Source 1: report.pdf (relevance: 0.85)]
   <chunk text>
   
   ---
   
   [Source 2: notes.md (relevance: 0.72)]
   <chunk text>
   ```
3. **Generate:** Sends the system prompt + context + conversation history + user query to the LLM.
4. **Source Citations:** Returns a trimmed version of the sources (first 200 chars each) for frontend citation cards.

**System prompt instructs the LLM to:**
- Use the reference documents to answer accurately
- Cite sources as `[Source N]`
- Say "I don't know" if documents don't help (avoids hallucination)

### 4.3 Summarization Agent

**File:** `agents/summarize_agent.py`

Specialized agent for condensing text. Uses a lower temperature (0.3) for more focused, deterministic output.

**Structured output format:**
1. Overview (1-2 sentence high-level summary)
2. Key Points (bullet list)
3. Details (critical specifics)
4. Conclusion (takeaway)

### 4.4 Code Agent

**File:** `agents/code_agent.py`

Specialized for programming queries. Uses the lowest temperature (0.2) and higher max tokens (1500) for detailed code analysis.

**Structured approach:**
1. Identify the language and framework
2. Explain what the code does
3. Point out issues (bugs, anti-patterns, security)
4. Suggest improvements with corrected code
5. For errors: explain WHY, provide fix, suggest prevention

### 4.5 Agent Router — LLM-Based Intent Detection

**File:** `agents/router.py` — Class `AgentRouter`

The router is the **brain** of the multi-agent system. It uses the LLM itself to classify which agent should handle a query.

#### Intent Classification

```python
INTENT_SYSTEM_PROMPT = """You are an intent classifier. Classify the user's message into exactly one category.

Categories:
- "rag": Questions about uploaded documents
- "summarize": Requests to summarize text
- "code": Code-related questions
- "general": General conversation

Respond with ONLY a JSON object: {"intent": "<category>", "confidence": <0.0-1.0>}"""
```

**How it works:**
1. The user's query is sent to `gpt-4o-mini` with `temperature=0.0` (deterministic) and `max_tokens=50` (minimal).
2. The LLM returns a JSON object like `{"intent": "rag", "confidence": 0.92}`.
3. The router selects the corresponding agent from its registry.
4. If classification fails (JSON parse error, API error), it **defaults to the RAG agent** as a safe fallback.

#### Routing Flow

```
User Query
     │
     ▼
  Intent Classification (LLM call, temp=0.0)
     │
     ├─ "rag"       → RAG Agent (hybrid search + doc Q&A)
     ├─ "summarize" → Summarize Agent (structured summary)
     ├─ "code"      → Code Agent (code analysis)
     └─ "general"   → RAG Agent (fallback, handles no-context gracefully)
```

#### Prompt Chaining

The router also supports **sequential prompt chaining** via `chain_prompts()`:
```python
results = await agent_router.chain_prompts([
    "Summarize this document",
    "Based on the summary, identify key risks",
    "Suggest mitigations for each risk"
])
```

Each step's output is fed as context into the next step's prompt, enabling multi-step reasoning workflows.

---

## 5. Conversation Memory

The project implements conversation memory at **two levels**:

### Level 1: Agent Router (Chat History Injection)

**File:** `agents/router.py`

```python
self.conversation_history: List[Dict[str, str]] = []
```

- Every user message and assistant response is appended to `conversation_history`.
- This list is passed as `context` to whichever agent processes the query.
- **Window size:** Last 20 messages (pruned automatically).
- Enables the agent to understand follow-up questions like "tell me more about that" or "what about the second point?"

### Level 2: LangChain ConversationBufferWindowMemory

**File:** `services/llm_service.py`

```python
self.memory = ConversationBufferWindowMemory(
    k=10,                # Keep last 10 exchanges
    return_messages=True,
    memory_key="history"
)
```

- Used by the `LLMService` (for workflow engine steps).
- Stores the last 10 input/output exchanges.
- Automatically injects history into prompts via LangChain's message system.

### Per-Agent History Injection

Each agent (RAG, Summarize, Code) includes the **last 6 messages** from the conversation history in its prompt:
```python
if history:
    messages.extend(history[-6:])
```

This ensures contextual continuity without overwhelming the LLM's context window.

---

## 6. Generation — Prompt Construction & LLM Calling

### LLM Service (GitHub Models)

**File:** `services/github_models.py`

Uses the **OpenAI Python SDK** pointed at GitHub's Models API endpoint:

```python
self.client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=settings.GITHUB_TOKEN
)
```

| Capability | Model | Usage |
|------------|-------|-------|
| **Chat Completion** | `gpt-4o-mini` | All generation tasks (Q&A, summarization, code analysis, intent classification, evaluation) |
| **Embeddings** | `text-embedding-3-small` | Document chunk embedding + query embedding |

**Reliability features:**
- **Retry with exponential backoff:** Up to 3 attempts with delays of 1s, 2s, 4s.
- **Streaming support:** `chat_completion_stream()` yields token-by-token output.

### RAG Agent Prompt Structure

The RAG agent builds its messages list as:

```
[
  { "role": "system",    "content": "<agent description> + <reference documents>" },
  { "role": "user",      "content": "<historical msg 1>" },   # ┐
  { "role": "assistant", "content": "<historical msg 2>" },   # │ Last 6 history msgs
  ...                                                          # ┘
  { "role": "user",      "content": "<current query>" }
]
```

---

## 7. Streaming (Server-Sent Events)

**File:** `api/endpoints/chat.py` — endpoint `/chat/stream`

The streaming endpoint uses **Server-Sent Events (SSE)** for real-time response delivery:

```python
@router.post("/chat/stream")
async def chat_stream(request):
    def generate():
        for chunk in agent_router.route_stream(request.message):
            if chunk.startswith("__META__"):
                # Agent metadata (which agent, intent, confidence)
                yield f"data: {json.dumps({'metadata': ...})}\n\n"
            else:
                # Response text chunk
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

**Stream protocol:**
1. **First event:** Agent metadata (agent name, intent, confidence score)
2. **Subsequent events:** Response text chunks (token by token)
3. **Final event:** `[DONE]` signal

This allows the frontend to:
- Show which agent is responding before text appears
- Display text token-by-token for a "typing" effect
- Know when the response is complete

---

## 8. Guardrails — Security & Safety Layer

**File:** `services/guardrails.py` — Class `GuardrailsService`

A three-layer security system applied to every request.

### 8.1 Prompt Injection Detection

Regex-based detection of known prompt injection patterns:

```python
INJECTION_PATTERNS = [
    r"ignore (?:all )?(?:previous |above )?instructions",
    r"disregard (?:all )?(?:previous |above )?(?:instructions|prompts)",
    r"you are now",
    r"new instructions:",
    r"system prompt:",
    r"override (?:your |the )?(?:system|instructions|rules)",
    r"pretend (?:you are|to be)",
    r"act as (?:if|though)",
    r"forget (?:everything|all|your)",
    r"do not follow",
    r"reveal (?:your|the) (?:system|prompt|instructions)",
    r"what (?:is|are) your (?:system|instructions|prompt)",
]
```

Also enforces a **10,000 character message length limit** to prevent abuse via extremely long prompts.

### 8.2 PII Redaction

Output sanitization that redacts personally identifiable information:

| PII Type | Pattern | Replacement |
|----------|---------|-------------|
| Email | `user@example.com` | `[REDACTED-EMAIL]` |
| Phone | `(123) 456-7890` | `[REDACTED-PHONE]` |
| SSN | `123-45-6789` | `[REDACTED-SSN]` |
| Credit Card | `1234 5678 9012 3456` | `[REDACTED-CREDIT_CARD]` |

### 8.3 Rate Limiting

**Token-bucket algorithm** per client:
- **Window:** 60 seconds (sliding)
- **Limit:** 30 requests per minute per client (configurable)
- Old entries are pruned each time a new request arrives
- Returns a user-friendly error message when exceeded

### Request Validation Pipeline

```python
def validate_request(text, client_id):
    1. Check rate limit    → block if exceeded
    2. Check for injection → block if detected
    3. Return (True, None) → allow request through
```

Every `/chat` and `/chat/stream` request passes through this pipeline before reaching any agent.

---

## 9. RAG Evaluation — LLM-as-Judge

**File:** `services/evaluation.py` — Class `EvaluationService`

The project includes a built-in evaluation system that uses the LLM itself as a judge to score RAG response quality across three dimensions:

### Faithfulness Score (0.0 → 1.0)

> "Is the answer grounded in the source documents?"

- 1.0 = Every claim is directly supported by sources
- 0.5 = Some claims supported, some not
- 0.0 = Answer contradicts or has no basis in sources

### Relevance Score (0.0 → 1.0)

> "Are the retrieved sources relevant to the query?"

- 1.0 = Sources directly address the question with perfect information
- 0.5 = Sources are somewhat related but don't fully address the question
- 0.0 = Sources are completely irrelevant

### Answer Quality Score (0.0 → 1.0)

> "Overall response accuracy and clarity?"

Considers: accuracy, completeness, clarity, organization, and helpfulness.

### Overall Score

```python
overall = (faithfulness + relevance + quality) / 3
```

Each metric is evaluated via a separate LLM call with `temperature=0.0` (deterministic) and structured JSON output. The endpoint `POST /evaluate` allows the frontend to request evaluation of any Q&A pair.

---

## 10. Workflow Engine — Multi-Step Prompt Chaining

**File:** `services/workflow_engine.py` — Class `WorkflowEngine`

Executes multi-step AI workflows where each step's output feeds into the next:

```python
async def execute_workflow(self, execution, workflow):
    context_data = {}
    for step in workflow.steps_definition:
        # Replace {placeholders} with outputs from previous steps
        formatted_prompt = step["prompt"].format(**context_data)
        output = await llm_service.process_step(step["type"], formatted_prompt)
        context_data[step["id"]] = output
```

**Step types supported:**
- `generation` — open-ended generation
- `summary` — text summarization
- `analysis` — structured data analysis
- `code_review` — code review with bug detection

**Features:**
- Placeholder substitution (`{step_id}` references previous step outputs)
- Per-step logging with timestamps
- Error handling with graceful failure states
- Execution tracking (status: running → completed/failed)

---

## 11. Configuration & Tunables

**File:** `core/config.py`

All parameters are configurable via environment variables or `.env` file:

| Variable | Default | What It Controls |
|----------|---------|------------------|
| `CHUNK_SIZE` | 500 | Maximum characters per text chunk |
| `CHUNK_OVERLAP` | 50 | Characters of overlap between consecutive chunks |
| `TOP_K_RESULTS` | 5 | Number of search results returned |
| `BM25_WEIGHT` | 0.3 | BM25 weight in hybrid search (semantic = 1 - this) |
| `CHAT_MODEL` | `gpt-4o-mini` | LLM for generation and classification |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model for vector embeddings |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | ChromaDB persistent storage directory |
| `GUARDRAILS_ENABLED` | `true` | Enable/disable the security layer |
| `MAX_REQUESTS_PER_MINUTE` | 30 | Rate limit per client |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## 12. End-to-End Data Flow (Putting It All Together)

### Document Upload Flow

```
User uploads PDF
       │
       ▼
  document_parser.parse()          ← Extract text from PDF (with page markers)
       │
       ▼
  text_splitter.split_text()       ← RecursiveCharacterTextSplitter: 500 chars, 50 overlap
       │
       ▼
  github_models.get_embeddings()   ← Batch embed all chunks via text-embedding-3-small
       │
       ├──► ChromaDB.add()         ← Store (id, vector, text, metadata)
       └──► BM25Index.add()        ← Store (text, tokens, metadata)
```

### Query Flow

```
User asks: "What does the report say about Q4 revenue?"
       │
       ▼
  guardrails.validate_request()    ← Rate limit check + injection detection
       │
       ▼
  router._classify_intent()        ← LLM classifies as "rag" (confidence: 0.95)
       │
       ▼
  rag_agent.process()
       │
       ├── vector_store.hybrid_search()
       │       │
       │       ├── Semantic: embed query → HNSW search → top 10 by cosine
       │       ├── BM25: tokenize query → score all docs → top 10 by BM25
       │       └── RRF: merge ranks (70% semantic, 30% BM25) → top 5
       │
       ├── _build_context()        ← Format chunks as labeled reference docs
       ├── _build_messages()       ← System prompt + history + context + query
       └── github_models.chat_completion()
               │
               ▼
         "According to [Source 1], Q4 revenue was $50M, representing
          a 15% increase YoY. [Source 3] notes that..."
               │
               ▼
          Return {response, sources, agent_used, intent}
```

---

This document covers every technical strategy and mechanism used in the AI Workflow Copilot project. Each component is designed to work together in a modular, production-grade pipeline that balances retrieval accuracy, generation quality, system safety, and user experience.
