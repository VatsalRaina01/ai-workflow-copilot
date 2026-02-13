# Setup Guide - AI Workflow Copilot

## Prerequisites

1. **Python 3.10+** - [Download](https://www.python.org/downloads/)
2. **Node.js 18+** - [Download](https://nodejs.org/)
3. **GitHub Personal Access Token**

## Getting Your GitHub Token

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name like "AI Workflow Copilot"
4. **IMPORTANT**: Check the box for `repo` scope (this includes `models:read`)
5. Click "Generate token"
6. **Copy the token immediately** (you won't see it again!)

## Installation

### Option 1: Quick Start (Windows)

1. **Set your GitHub token:**
```cmd
set GITHUB_TOKEN=ghp_your_token_here
```

2. **Start the backend:**
```cmd
start-backend.bat
```

3. **In a new terminal, start the frontend:**
```cmd
start-frontend.bat
```

4. **Open your browser:** http://localhost:5173

### Option 2: Manual Setup

#### Backend Setup

```cmd
cd backend

REM Create .env file
copy .env.example .env
REM Edit .env and add your GITHUB_TOKEN

REM Install dependencies
pip install -r requirements.txt

REM Start server
python -m uvicorn app.main:app --reload --port 8000
```

#### Frontend Setup

```cmd
cd frontend

REM Install dependencies
npm install

REM Start dev server
npm run dev
```

## Verification

### 1. Check Backend Health

Open http://localhost:8000/api/v1/health

You should see:
```json
{
  "status": "healthy",
  "chat_model": "gpt-4o-mini",
  "embedding_model": "text-embedding-3-small"
}
```

### 2. Test the UI

1. Go to http://localhost:5173
2. You'll see the chat interface
3. Try asking a question (it will work even without documents, but won't have context)

### 3. Test RAG Pipeline

1. Create a test file `test.txt`:
```
The company vacation policy states that employees receive 15 days of paid time off per year.
New employees must complete a 90-day probation period before using vacation days.
```

2. Upload it via the UI (⬆️ Upload Doc button)
3. Ask: "How many vacation days do employees get?"
4. The AI should respond with information from your document!

## Troubleshooting

### "401 Unauthorized" or GitHub API errors

- Check your `GITHUB_TOKEN` is set correctly
- Verify the token has `repo` scope
- Make sure you're using a Classic token, not Fine-grained

### "Module not found" errors

```cmd
cd backend
pip install -r requirements.txt --upgrade
```

### Frontend won't start

```cmd
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### ChromaDB errors

The app uses an in-memory ChromaDB (no setup needed). If you see SQLite errors, update:
```cmd
pip install chromadb --upgrade
```

## Next Steps

- Upload your own documents (.txt or .md files)
- Ask domain-specific questions
- Explore the API docs at http://localhost:8000/api/v1/docs
- Check the agent orchestration in `backend/app/agents/`

## Production Deployment

For production:
1. Use persistent ChromaDB (update `vector_store.py`)
2. Set up proper environment variables
3. Use production WSGI server (Gunicorn)
4. Build frontend: `npm run build`
5. Deploy static files + API separately

See README.md for architecture details.
