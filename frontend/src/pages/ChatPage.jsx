import React, { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { api } from '../services/api';

// Agent badge config
const AGENT_CONFIG = {
    rag: { icon: '📚', label: 'RAG', color: '#6366f1' },
    summarize: { icon: '📝', label: 'Summarize', color: '#22c55e' },
    code: { icon: '💻', label: 'Code', color: '#f59e0b' },
    general: { icon: '💬', label: 'General', color: '#8b5cf6' },
    guardrails: { icon: '🛡️', label: 'Guardrails', color: '#ef4444' },
};

// File type config
const FILE_ICONS = {
    '.pdf': '📕',
    '.docx': '📘',
    '.doc': '📘',
    '.txt': '📄',
    '.md': '📑',
};

const ChatPage = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [docCount, setDocCount] = useState(0);
    const [uploadStatus, setUploadStatus] = useState('');
    const [isDragging, setIsDragging] = useState(false);
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [conversations, setConversations] = useState([{ id: 1, title: 'New Chat', active: true }]);
    const messagesEndRef = useRef(null);
    const fileInputRef = useRef(null);
    const dropZoneRef = useRef(null);

    useEffect(() => { loadDocCount(); }, []);
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const loadDocCount = async () => {
        try {
            const data = await api.getDocumentCount();
            setDocCount(data.count);
        } catch (e) { console.error('Failed to load doc count'); }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMessage = input.trim();
        setInput('');
        setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
        setIsLoading(true);

        // Add placeholder for assistant response
        setMessages(prev => [...prev, {
            role: 'assistant', content: '', isStreaming: true,
            agent: null, sources: []
        }]);

        try {
            await api.chatStream(
                userMessage,
                // On chunk
                (chunk) => {
                    setMessages(prev => {
                        const updated = [...prev];
                        const last = updated.length - 1;
                        updated[last] = { ...updated[last], content: updated[last].content + chunk };
                        return updated;
                    });
                },
                // On metadata
                (metadata) => {
                    setMessages(prev => {
                        const updated = [...prev];
                        const last = updated.length - 1;
                        updated[last] = { ...updated[last], agent: metadata.agent_used, agentName: metadata.agent_name };
                        return updated;
                    });
                }
            );

            // Mark streaming complete
            setMessages(prev => {
                const updated = [...prev];
                updated[updated.length - 1].isStreaming = false;
                return updated;
            });
        } catch (error) {
            setMessages(prev => {
                const updated = [...prev];
                updated[updated.length - 1] = {
                    role: 'assistant',
                    content: 'Sorry, there was an error processing your request.',
                    isError: true
                };
                return updated;
            });
        } finally {
            setIsLoading(false);
        }
    };

    // ─── File Upload Handlers ─────────────────────────────────────

    const processFiles = async (files) => {
        if (!files.length) return;
        const fileArray = Array.from(files);
        setUploadStatus(`Uploading ${fileArray.length} file(s)...`);

        try {
            const results = await api.uploadMultipleDocuments(fileArray);
            const successCount = results.filter(r => r.success).length;
            const failCount = results.filter(r => !r.success).length;

            let status = `✅ ${successCount} file(s) uploaded`;
            if (failCount > 0) status += ` | ❌ ${failCount} failed`;
            setUploadStatus(status);
            await loadDocCount();
            setTimeout(() => setUploadStatus(''), 4000);
        } catch (error) {
            setUploadStatus(`❌ ${error.message}`);
        }
    };

    const handleFileUpload = (e) => {
        processFiles(e.target.files);
        e.target.value = '';
    };

    // Drag & Drop
    const handleDragEnter = useCallback((e) => { e.preventDefault(); setIsDragging(true); }, []);
    const handleDragLeave = useCallback((e) => { e.preventDefault(); setIsDragging(false); }, []);
    const handleDragOver = useCallback((e) => { e.preventDefault(); }, []);
    const handleDrop = useCallback((e) => {
        e.preventDefault();
        setIsDragging(false);
        processFiles(e.dataTransfer.files);
    }, []);

    // ─── Conversation Sidebar ────────────────────────────────────

    const newConversation = () => {
        setMessages([]);
        api.clearConversation();
        const newId = Date.now();
        setConversations(prev => [
            { id: newId, title: 'New Chat', active: true },
            ...prev.map(c => ({ ...c, active: false }))
        ]);
    };

    const clearChat = () => {
        setMessages([]);
        api.clearConversation();
    };

    // ─── Markdown Components ─────────────────────────────────────

    const markdownComponents = {
        code({ node, inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '');
            return !inline && match ? (
                <SyntaxHighlighter
                    style={oneDark}
                    language={match[1]}
                    PreTag="div"
                    customStyle={{ borderRadius: '8px', fontSize: '13px', margin: '8px 0' }}
                    {...props}
                >
                    {String(children).replace(/\n$/, '')}
                </SyntaxHighlighter>
            ) : (
                <code className="inline-code" {...props}>{children}</code>
            );
        },
        table({ children }) { return <div className="table-wrapper"><table>{children}</table></div>; },
        a({ href, children }) { return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>; },
    };

    // ─── Source Citation Component ────────────────────────────────

    const SourceCards = ({ sources }) => {
        const [expanded, setExpanded] = useState(false);
        if (!sources || sources.length === 0) return null;

        return (
            <div className="source-cards">
                <button className="source-toggle" onClick={() => setExpanded(!expanded)}>
                    📎 {sources.length} source{sources.length > 1 ? 's' : ''} used
                    <span className={`chevron ${expanded ? 'open' : ''}`}>▾</span>
                </button>
                {expanded && (
                    <div className="source-list">
                        {sources.map((src, i) => (
                            <div key={i} className="source-card">
                                <div className="source-header">
                                    <span className="source-name">
                                        {FILE_ICONS[src.file_type] || '📄'} {src.filename}
                                    </span>
                                    <span className="source-score">
                                        {(src.score * 100).toFixed(0)}% match
                                    </span>
                                </div>
                                <div className="source-preview">{src.content}</div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        );
    };

    // ─── Render ──────────────────────────────────────────────────

    return (
        <div
            className={`app-layout ${isDragging ? 'dragging' : ''}`}
            onDragEnter={handleDragEnter}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
        >
            {/* Sidebar */}
            <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
                <div className="sidebar-header">
                    <h2>Conversations</h2>
                    <button className="btn-new-chat" onClick={newConversation}>+ New Chat</button>
                </div>
                <div className="conversation-list">
                    {conversations.map(conv => (
                        <div key={conv.id} className={`conv-item ${conv.active ? 'active' : ''}`}>
                            💬 {conv.title}
                        </div>
                    ))}
                </div>
            </aside>

            {/* Main Chat */}
            <div className="chat-container">
                {/* Header */}
                <header className="chat-header">
                    <div className="header-content">
                        <div className="logo">
                            <button className="btn-sidebar" onClick={() => setSidebarOpen(!sidebarOpen)}>☰</button>
                            <span className="logo-icon">🤖</span>
                            <h1>AI Workflow Copilot</h1>
                            <span className="version-badge">v3.0</span>
                        </div>
                        <div className="header-actions">
                            <div className="doc-badge" title="Indexed document chunks">
                                📄 {docCount} chunks
                            </div>
                            <button className="btn-upload" onClick={() => fileInputRef.current?.click()}>
                                ⬆️ Upload
                            </button>
                            <button className="btn-clear" onClick={clearChat}>🗑️ Clear</button>
                        </div>
                    </div>
                    {uploadStatus && <div className="upload-status">{uploadStatus}</div>}
                    <input
                        type="file"
                        ref={fileInputRef}
                        onChange={handleFileUpload}
                        accept=".txt,.md,.pdf,.docx"
                        multiple
                        style={{ display: 'none' }}
                    />
                </header>

                {/* Drag & Drop Overlay */}
                {isDragging && (
                    <div className="drop-overlay">
                        <div className="drop-content">
                            <span className="drop-icon">📁</span>
                            <h3>Drop files here</h3>
                            <p>Supports .txt, .md, .pdf, .docx</p>
                        </div>
                    </div>
                )}

                {/* Messages */}
                <main className="messages-container">
                    {messages.length === 0 ? (
                        <div className="empty-state">
                            <div className="empty-icon">💬</div>
                            <h2>Welcome to AI Workflow Copilot</h2>
                            <p>Upload documents and ask questions. The AI uses multi-agent routing with RAG.</p>
                            <div className="feature-cards">
                                <div className="feature-card">
                                    <span>📚</span>
                                    <strong>RAG Search</strong>
                                    <small>Document Q&A</small>
                                </div>
                                <div className="feature-card">
                                    <span>📝</span>
                                    <strong>Summarize</strong>
                                    <small>Condense text</small>
                                </div>
                                <div className="feature-card">
                                    <span>💻</span>
                                    <strong>Code Help</strong>
                                    <small>Debug & review</small>
                                </div>
                                <div className="feature-card">
                                    <span>📕</span>
                                    <strong>Multi-Format</strong>
                                    <small>PDF, DOCX, TXT</small>
                                </div>
                            </div>
                            <p className="hint">Drag & drop files or click Upload to get started.</p>
                        </div>
                    ) : (
                        messages.map((msg, idx) => (
                            <div key={idx} className={`message ${msg.role}`}>
                                <div className="message-avatar">
                                    {msg.role === 'user' ? '👤' : '🤖'}
                                </div>
                                <div className="message-body">
                                    {/* Agent badge */}
                                    {msg.role === 'assistant' && msg.agent && (
                                        <div className="agent-badge" style={{
                                            borderColor: AGENT_CONFIG[msg.agent]?.color || '#8b5cf6'
                                        }}>
                                            {AGENT_CONFIG[msg.agent]?.icon} {AGENT_CONFIG[msg.agent]?.label || msg.agent}
                                        </div>
                                    )}

                                    <div className={`message-content ${msg.isError ? 'error' : ''}`}>
                                        {msg.role === 'assistant' && msg.content ? (
                                            <ReactMarkdown
                                                remarkPlugins={[remarkGfm]}
                                                components={markdownComponents}
                                            >
                                                {msg.content}
                                            </ReactMarkdown>
                                        ) : msg.role === 'user' ? (
                                            msg.content
                                        ) : (
                                            msg.isStreaming && <span className="typing">▍</span>
                                        )}
                                    </div>

                                    {/* Source citations */}
                                    {msg.sources && msg.sources.length > 0 && (
                                        <SourceCards sources={msg.sources} />
                                    )}
                                </div>
                            </div>
                        ))
                    )}
                    <div ref={messagesEndRef} />
                </main>

                {/* Input */}
                <footer className="input-container">
                    <form onSubmit={handleSubmit} className="input-form">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Ask about documents, summarize text, or get code help..."
                            disabled={isLoading}
                            className="chat-input"
                        />
                        <button type="submit" disabled={isLoading || !input.trim()} className="btn-send">
                            {isLoading ? '⏳' : '➤'}
                        </button>
                    </form>
                    <div className="input-hint">
                        Multi-agent AI • Hybrid Search • PDF/DOCX Support • Guardrails Enabled
                    </div>
                </footer>
            </div>
        </div>
    );
};

export default ChatPage;
