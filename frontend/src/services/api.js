const API_BASE = 'http://localhost:8000/api/v1';

export const api = {
    /**
     * Send a chat message and get response
     */
    async chat(message) {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, stream: false }),
        });

        if (!response.ok) {
            throw new Error('Chat request failed');
        }

        return response.json();
    },

    /**
     * Send a chat message with streaming response + agent metadata
     */
    async chatStream(message, onChunk, onMetadata) {
        const response = await fetch(`${API_BASE}/chat/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message }),
        });

        if (!response.ok) {
            throw new Error('Stream request failed');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const text = decoder.decode(value);
            const lines = text.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') {
                        return;
                    }
                    try {
                        const parsed = JSON.parse(data);
                        if (parsed.metadata && onMetadata) {
                            onMetadata(parsed.metadata);
                        } else if (parsed.chunk) {
                            onChunk(parsed.chunk);
                        }
                    } catch (e) {
                        // Ignore parse errors
                    }
                }
            }
        }
    },

    /**
     * Upload a document for RAG indexing (supports txt, md, pdf, docx)
     */
    async uploadDocument(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE}/documents`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }

        return response.json();
    },

    /**
     * Upload multiple documents
     */
    async uploadMultipleDocuments(files) {
        const results = [];
        for (const file of files) {
            try {
                const result = await this.uploadDocument(file);
                results.push({ file: file.name, success: true, ...result });
            } catch (error) {
                results.push({ file: file.name, success: false, error: error.message });
            }
        }
        return results;
    },

    /**
     * Get document count
     */
    async getDocumentCount() {
        const response = await fetch(`${API_BASE}/documents/count`);
        return response.json();
    },

    /**
     * Clear conversation history
     */
    async clearConversation() {
        const response = await fetch(`${API_BASE}/clear`, { method: 'POST' });
        return response.json();
    },

    /**
     * Get available agents
     */
    async getAgents() {
        const response = await fetch(`${API_BASE}/agents`);
        return response.json();
    },

    /**
     * Evaluate a RAG response
     */
    async evaluate(question, answer, sources) {
        const response = await fetch(`${API_BASE}/evaluate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, answer, sources }),
        });
        return response.json();
    },

    /**
     * Health check
     */
    async healthCheck() {
        const response = await fetch(`${API_BASE}/health`);
        return response.json();
    }
};
