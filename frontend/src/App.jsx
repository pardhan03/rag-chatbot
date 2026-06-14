import { useState, useEffect, useRef } from 'react';
import axios from 'axios';

// API Base URL
const API_URL = 'http://localhost:8000';

function App() {
  // Navigation
  const [activeTab, setActiveTab] = useState('chat'); // 'chat' | 'ingest'

  // Model & Search Settings (Stored in state)
  const [modelProvider, setModelProvider] = useState('ollama'); // 'ollama' | 'openai'
  const [modelName, setModelName] = useState('llama3.2'); // 'llama3.2' | 'gpt-4o'
  const [embeddingProvider, setEmbeddingProvider] = useState('ollama');
  const [embeddingModel, setEmbeddingModel] = useState('nomic-embed-text');
  const [searchType, setSearchType] = useState('similarity'); // 'similarity' | 'similarity_score_threshold' | 'mmr'
  const [k, setK] = useState(4);
  const [scoreThreshold, setScoreThreshold] = useState(0.35);
  const [openaiApiKey, setOpenaiApiKey] = useState('');

  // DB Stats
  const [dbStats, setDbStats] = useState({
    status: 'empty',
    total_chunks: 0,
    collection_name: null,
    documents_represented: []
  });

  // Chat State
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState(''); // 'rewriting' | 'searching' | 'generating' | ''
  const [rewrittenQuery, setRewrittenQuery] = useState(null);

  // Document Management State
  const [documents, setDocuments] = useState([]);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  // Ingestion State
  const [splitterType, setSplitterType] = useState('recursive'); // 'character' | 'recursive' | 'semantic' | 'agentic'
  const [chunkSize, setChunkSize] = useState(1500);
  const [chunkOverlap, setChunkOverlap] = useState(200);
  const [breakpointThresholdType, setBreakpointThresholdType] = useState('percentile');
  const [breakpointThresholdAmount, setBreakpointThresholdAmount] = useState(70);
  const [ingestStatus, setIngestStatus] = useState({
    status: 'idle',
    message: 'No active ingestion',
    logs: [],
    total_chunks: 0
  });

  const chatEndRef = useRef(null);
  const terminalEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Load API key from local storage on mount
  useEffect(() => {
    const savedKey = localStorage.getItem('openai_api_key');
    if (savedKey) {
      setOpenaiApiKey(savedKey);
    }
    fetchDbStats();
    fetchDocuments();
    fetchIngestionStatus();
  }, []);

  // Update default models when provider changes
  useEffect(() => {
    if (modelProvider === 'openai') {
      setModelName('gpt-4o');
    } else {
      setModelName('llama3.2');
    }
  }, [modelProvider]);

  useEffect(() => {
    if (embeddingProvider === 'openai') {
      setEmbeddingModel('text-embedding-3-small');
    } else {
      setEmbeddingModel('nomic-embed-text');
    }
  }, [embeddingProvider]);

  // Save API key when it changes
  const handleApiKeyChange = (e) => {
    const key = e.target.value;
    setOpenaiApiKey(key);
    localStorage.setItem('openai_api_key', key);
  };

  // Scroll to bottom of chat
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, chatLoading, currentStep]);

  // Scroll to bottom of terminal when logs update
  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [ingestStatus.logs]);

  // Polling for Ingestion status when processing
  useEffect(() => {
    let interval;
    if (ingestStatus.status === 'processing') {
      interval = setInterval(() => {
        fetchIngestionStatus();
        fetchDbStats();
      }, 1500);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [ingestStatus.status]);

  // API Call: DB Stats
  const fetchDbStats = async () => {
    try {
      const res = await axios.get(`${API_URL}/api/db/stats`, {
        params: {
          embedding_provider: embeddingProvider,
          embedding_model: embeddingModel,
          openai_api_key: openaiApiKey || undefined
        }
      });
      setDbStats(res.data);
    } catch (err) {
      console.error('Error fetching DB stats:', err);
    }
  };

  // API Call: List Documents
  const fetchDocuments = async () => {
    try {
      const res = await axios.get(`${API_URL}/api/docs`);
      setDocuments(res.data);
    } catch (err) {
      console.error('Error listing documents:', err);
    }
  };

  // API Call: Get Ingestion Status
  const fetchIngestionStatus = async () => {
    try {
      const res = await axios.get(`${API_URL}/api/ingest/status`);
      setIngestStatus(res.data);
      if (res.data.status === 'completed' || res.data.status === 'error') {
        fetchDbStats();
      }
    } catch (err) {
      console.error('Error fetching ingestion status:', err);
    }
  };

  // API Call: Clear DB
  const handleClearDb = async () => {
    if (!window.confirm('Are you sure you want to clear the vector database? All embeddings will be permanently deleted.')) return;
    try {
      await axios.post(`${API_URL}/api/db/clear`);
      fetchDbStats();
      setIngestStatus({
        status: 'idle',
        message: 'Vector database cleared',
        logs: [],
        total_chunks: 0
      });
      alert('Vector database cleared successfully.');
    } catch (err) {
      console.error('Error clearing database:', err);
      alert('Failed to clear database.');
    }
  };

  // API Call: Ingestion Start
  const handleStartIngest = async () => {
    try {
      setIngestStatus(prev => ({
        ...prev,
        status: 'processing',
        logs: ['[INFO] Requesting ingestion pipeline initialization...']
      }));
      
      const payload = {
        splitter_type: splitterType,
        chunk_size: parseInt(chunkSize),
        chunk_overlap: parseInt(chunkOverlap),
        breakpoint_threshold_type: breakpointThresholdType,
        breakpoint_threshold_amount: parseInt(breakpointThresholdAmount),
        model_provider: modelProvider,
        model_name: modelName,
        embedding_provider: embeddingProvider,
        embedding_model: embeddingModel,
        openai_api_key: openaiApiKey || null
      };

      await axios.post(`${API_URL}/api/ingest`, payload);
      fetchIngestionStatus();
    } catch (err) {
      console.error('Ingestion initiation failed:', err);
      const errMsg = err.response?.data?.detail || err.message;
      setIngestStatus(prev => ({
        ...prev,
        status: 'error',
        logs: [...prev.logs, `[ERROR] Ingestion failed to trigger: ${errMsg}`]
      }));
    }
  };

  // API Call: Upload Document
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.name.endsWith('.txt')) {
      setUploadError('Only .txt files are supported.');
      return;
    }

    setUploadLoading(true);
    setUploadError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      await axios.post(`${API_URL}/api/docs/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      fetchDocuments();
    } catch (err) {
      console.error('Upload failed:', err);
      setUploadError(err.response?.data?.detail || 'Upload failed.');
    } finally {
      setUploadLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // API Call: Delete Document
  const handleDeleteDoc = async (filename) => {
    if (!window.confirm(`Delete document "${filename}"?`)) return;
    try {
      await axios.delete(`${API_URL}/api/docs/${filename}`);
      fetchDocuments();
    } catch (err) {
      console.error('Delete failed:', err);
      alert('Failed to delete file.');
    }
  };

  // API Call: Send Message
  const handleSendMessage = async (e) => {
    if (e) e.preventDefault();
    if (!chatInput.trim() || chatLoading) return;

    const userQuery = chatInput;
    setChatInput('');
    setChatLoading(true);
    setRewrittenQuery(null);

    // Append user message immediately
    const userMessage = {
      role: 'user',
      content: userQuery,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    setMessages(prev => [...prev, userMessage]);

    // Simulate RAG pipeline steps
    setCurrentStep('rewriting');
    
    // Prepare API history format
    const apiHistory = messages.map(msg => ({
      role: msg.role,
      content: msg.content
    }));

    try {
      const payload = {
        question: userQuery,
        history: apiHistory,
        model_provider: modelProvider,
        model_name: modelName,
        embedding_provider: embeddingProvider,
        embedding_model: embeddingModel,
        search_type: searchType,
        k: parseInt(k),
        score_threshold: parseFloat(scoreThreshold),
        openai_api_key: openaiApiKey || null
      };

      // Call Chat Endpoint
      const res = await axios.post(`${API_URL}/api/chat`, payload);
      
      const botMessage = {
        role: 'assistant',
        content: res.data.answer,
        rewritten_query: res.data.rewritten_query,
        chunks: res.data.chunks,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        showSources: false
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (err) {
      console.error('Chat failed:', err);
      const errMsg = err.response?.data?.detail || err.message;
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: `Error occurred: ${errMsg}. Make sure your backend server is running and models are loaded in Ollama or API key is valid.`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          isError: true
        }
      ]);
    } finally {
      setChatLoading(false);
      setCurrentStep('');
    }
  };

  const toggleSources = (index) => {
    setMessages(prev => prev.map((msg, i) => {
      if (i === index) {
        return { ...msg, showSources: !msg.showSources };
      }
      return msg;
    }));
  };

  // Helper formatting for bytes
  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="app-container">
      {/* Sidebar - Control Panel */}
      <aside className="sidebar">
        <div className="logo-container">
          <div className="logo-icon">R</div>
          <div className="logo-text">AuraRAG Studio</div>
        </div>

        <nav className="nav-tabs">
          <div 
            className={`nav-tab ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
            Chat Studio
          </div>
          <div 
            className={`nav-tab ${activeTab === 'ingest' ? 'active' : ''}`}
            onClick={() => setActiveTab('ingest')}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
            Ingest & Database
          </div>
        </nav>

        {/* Configurations Section */}
        <div className="sidebar-section">
          <div className="sidebar-title">Global Settings</div>
          
          {/* OpenAI Key */}
          <div className="form-group">
            <label>OpenAI API Key (Optional)</label>
            <input 
              type="password" 
              className="form-control"
              placeholder="sk-..."
              value={openaiApiKey}
              onChange={handleApiKeyChange}
            />
          </div>

          {/* Model Provider */}
          <div className="form-group">
            <label>LLM Provider</label>
            <select 
              className="form-control"
              value={modelProvider}
              onChange={(e) => setModelProvider(e.target.value)}
            >
              <option value="ollama">Ollama (Local)</option>
              <option value="openai">OpenAI (Cloud)</option>
            </select>
          </div>

          {/* Model Name */}
          <div className="form-group">
            <label>LLM Model</label>
            {modelProvider === 'ollama' ? (
              <select 
                className="form-control"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
              >
                <option value="llama3.2">llama3.2 (3B)</option>
                <option value="nemotron-3-super:cloud">nemotron-3-super:cloud</option>
              </select>
            ) : (
              <select 
                className="form-control"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
              >
                <option value="gpt-4o">gpt-4o</option>
                <option value="gpt-4o-mini">gpt-4o-mini</option>
                <option value="gpt-3.5-turbo">gpt-3.5-turbo</option>
              </select>
            )}
          </div>

          {/* Embedding Provider */}
          <div className="form-group">
            <label>Embedding Provider</label>
            <select 
              className="form-control"
              value={embeddingProvider}
              onChange={(e) => setEmbeddingProvider(e.target.value)}
            >
              <option value="ollama">Ollama (Local)</option>
              <option value="openai">OpenAI (Cloud)</option>
            </select>
          </div>

          {/* Embedding Model */}
          <div className="form-group">
            <label>Embedding Model</label>
            {embeddingProvider === 'ollama' ? (
              <select 
                className="form-control"
                value={embeddingModel}
                onChange={(e) => setEmbeddingModel(e.target.value)}
              >
                <option value="nomic-embed-text">nomic-embed-text</option>
              </select>
            ) : (
              <select 
                className="form-control"
                value={embeddingModel}
                onChange={(e) => setEmbeddingModel(e.target.value)}
              >
                <option value="text-embedding-3-small">text-embedding-3-small</option>
                <option value="text-embedding-3-large">text-embedding-3-large</option>
              </select>
            )}
          </div>

          {/* Search Strategy */}
          <div className="form-group">
            <label>Search Algorithm</label>
            <select 
              className="form-control"
              value={searchType}
              onChange={(e) => setSearchType(e.target.value)}
            >
              <option value="similarity">Similarity Search</option>
              <option value="similarity_score_threshold">Similarity Score Threshold</option>
              <option value="mmr">Max Marginal Relevance (MMR)</option>
            </select>
          </div>

          {/* Search K */}
          <div className="form-group">
            <label>Chunks to Retrieve (K)</label>
            <div className="slider-container">
              <input 
                type="range" 
                min="1" 
                max="10" 
                value={k}
                onChange={(e) => setK(parseInt(e.target.value))}
              />
              <span className="slider-value">{k}</span>
            </div>
          </div>

          {/* Score Threshold */}
          {searchType === 'similarity_score_threshold' && (
            <div className="form-group">
              <label>Relevance Threshold</label>
              <div className="slider-container">
                <input 
                  type="range" 
                  min="0.1" 
                  max="0.9" 
                  step="0.05"
                  value={scoreThreshold}
                  onChange={(e) => setScoreThreshold(parseFloat(e.target.value))}
                />
                <span className="slider-value">{scoreThreshold}</span>
              </div>
            </div>
          )}
        </div>

        {/* Database Status Widget */}
        <div style={{ marginTop: 'auto' }}>
          <div className="sidebar-title">Vector DB Status</div>
          <div className="db-status-card">
            <div className="db-status-row">
              <span className="db-status-label">Status</span>
              <span className="db-status-value">
                <span className={`badge ${ingestStatus.status}`}>
                  {ingestStatus.status}
                </span>
              </span>
            </div>
            <div className="db-status-row">
              <span className="db-status-label">Total Chunks</span>
              <span className="db-status-value">{dbStats.total_chunks || 0}</span>
            </div>
            <div className="db-status-row" style={{ flexDirection: 'column', gap: '0.2rem', marginTop: '0.5rem' }}>
              <span className="db-status-label">Ingested Files ({dbStats.documents_represented?.length || 0}):</span>
              <div style={{ maxHeight: '60px', overflowY: 'auto', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                {dbStats.documents_represented?.map(doc => (
                  <div key={doc} style={{ whiteSpace: 'nowrap', textOverflow: 'ellipsis', overflow: 'hidden' }}>• {doc}</div>
                ))}
                {(!dbStats.documents_represented || dbStats.documents_represented.length === 0) && (
                  <div>None</div>
                )}
              </div>
            </div>
            
            <button 
              className="btn btn-danger" 
              style={{ width: '100%', marginTop: '0.75rem', padding: '0.4rem', fontSize: '0.8rem' }}
              onClick={handleClearDb}
              disabled={ingestStatus.status === 'processing'}
            >
              Clear Vector Store
            </button>
          </div>
        </div>
      </aside>

      {/* Main Content Pane */}
      <main className="main-content">
        <header className="header-bar">
          <h2 className="header-title">
            {activeTab === 'chat' ? 'RAG Conversation Studio' : 'Data Ingestion & Pipeline Control'}
          </h2>
          <div className="header-actions">
            <button className="btn btn-secondary" onClick={fetchDbStats}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
              Refresh Stats
            </button>
          </div>
        </header>

        {activeTab === 'chat' ? (
          /* Tab 1: Chat interface */
          <div className="chat-container fade-in">
            {messages.length === 0 ? (
              <div className="empty-state">
                <div className="empty-state-icon">🤖</div>
                <h3>Welcome to AuraRAG Studio</h3>
                <p>
                  This is a locally-powered Retrieval-Augmented Generation chatbot. Ensure your vector database is populated with document chunks first.
                </p>
                <button className="btn btn-primary" onClick={() => setActiveTab('ingest')}>
                  Configure & Ingest Documents
                </button>
              </div>
            ) : (
              <div className="chat-messages">
                {messages.map((msg, i) => (
                  <div key={i} className={`chat-message-wrapper ${msg.role}`}>
                    <div className="chat-message">
                      {msg.content}
                      
                      {/* Query Rewriter Indicator */}
                      {msg.rewritten_query && (
                        <div className="rewritten-query-box">
                          <strong>Standalone search query:</strong> "{msg.rewritten_query}"
                        </div>
                      )}

                      {/* Sources Section */}
                      {msg.chunks && msg.chunks.length > 0 && (
                        <div>
                          <button 
                            className="source-toggle-btn"
                            onClick={() => toggleSources(i)}
                          >
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                              {msg.showSources ? (
                                <polyline points="18 15 12 9 6 15"/>
                              ) : (
                                <polyline points="6 9 12 15 18 9"/>
                              )}
                            </svg>
                            {msg.showSources ? 'Hide Sources' : `View ${msg.chunks.length} Sources`}
                          </button>
                          
                          {msg.showSources && (
                            <div className="sources-container">
                              {msg.chunks.map((chunk, ci) => (
                                <div key={ci} className="source-chunk-card">
                                  <div className="source-chunk-header">
                                    <span className="source-chunk-name">#{ci+1} {chunk.source}</span>
                                    <span className="source-chunk-score" style={{ color: chunk.score > 0.6 ? 'var(--success)' : 'var(--warning)' }}>
                                      Score: {(chunk.score * 100).toFixed(1)}%
                                    </span>
                                  </div>
                                  <div className="source-chunk-content">
                                    {chunk.content}
                                  </div>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                    <div className="chat-meta">
                      <span>{msg.role === 'user' ? 'You' : 'Assistant'}</span>
                      <span>•</span>
                      <span>{msg.timestamp}</span>
                    </div>
                  </div>
                ))}

                {/* Thinking states */}
                {chatLoading && (
                  <div className="chat-message-wrapper assistant">
                    <div className="chat-message" style={{ background: 'rgba(255, 255, 255, 0.02)', border: '1px dashed var(--border-color)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <div className="loading-dots">
                        <span></span><span></span><span></span>
                      </div>
                      <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                        {currentStep === 'rewriting' && 'Rewriting chat context query...'}
                        {currentStep === 'searching' && 'Performing similarity search on ChromaDB...'}
                        {currentStep === 'generating' && 'Synthesizing response from retrieved chunks...'}
                        {!currentStep && 'Thinking...'}
                      </span>
                    </div>
                  </div>
                )}
                
                <div ref={chatEndRef} />
              </div>
            )}

            {/* Input form */}
            <form onSubmit={handleSendMessage} className="chat-input-container">
              <div className="chat-input-wrapper">
                <input 
                  type="text" 
                  className="chat-input"
                  placeholder={dbStats.total_chunks === 0 ? "⚠️ Vector DB is empty. Run Ingestion first!" : "Ask a question about the documents..."}
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  disabled={chatLoading}
                />
                <button 
                  type="submit" 
                  className="btn btn-primary"
                  disabled={chatLoading || !chatInput.trim()}
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
                  Send
                </button>
              </div>
            </form>
          </div>
        ) : (
          /* Tab 2: Document Ingestion and Settings Manager */
          <div className="ingest-grid fade-in">
            {/* Left Card: Document Files List & Upload */}
            <div className="card">
              <h3 className="card-title">
                Document Repository
                <span style={{ fontSize: '0.75rem', fontWeight: 'normal', color: 'var(--text-muted)' }}>
                  {documents.length} files loaded
                </span>
              </h3>

              {/* Upload Dropzone */}
              <div className="upload-zone" onClick={() => fileInputRef.current.click()}>
                <input 
                  type="file" 
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  onChange={handleFileUpload}
                  accept=".txt"
                  disabled={uploadLoading}
                />
                <div className="upload-icon">📤</div>
                <div className="upload-text">
                  {uploadLoading ? 'Uploading and parsing document...' : 'Click to Upload Document'}
                </div>
                <div className="upload-subtext">Only .txt files are supported currently</div>
              </div>
              {uploadError && <div style={{ color: 'var(--danger)', fontSize: '0.8rem', marginBottom: '1rem' }}>{uploadError}</div>}

              {/* Files list */}
              <div className="doc-list">
                {documents.map((doc, idx) => (
                  <div key={idx} className="doc-item">
                    <div>
                      <div className="doc-name">{doc.name}</div>
                      <div className="doc-meta">
                        <span>{formatBytes(doc.size)}</span>
                        <span>•</span>
                        <span>{doc.words} words</span>
                        <span>•</span>
                        <span>{doc.chars} chars</span>
                      </div>
                    </div>
                    <button 
                      className="doc-delete-btn"
                      onClick={() => handleDeleteDoc(doc.name)}
                      title="Delete file"
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/></svg>
                    </button>
                  </div>
                ))}

                {documents.length === 0 && (
                  <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                    No documents uploaded. Copy documents to backend/docs or upload them here.
                  </div>
                )}
              </div>
            </div>

            {/* Right Card: Ingestion settings & pipeline runner */}
            <div className="card">
              <h3 className="card-title">Pipeline Configuration</h3>

              {/* Splitter select */}
              <div className="form-group">
                <label>Chunking Strategy</label>
                <select 
                  className="form-control"
                  value={splitterType}
                  onChange={(e) => setSplitterType(e.target.value)}
                >
                  <option value="recursive">Recursive Character Splitter</option>
                  <option value="character">Character Splitter</option>
                  <option value="semantic">Semantic Chunker (langchain-experimental)</option>
                  <option value="agentic">Agentic Chunker (LLM-based)</option>
                </select>
              </div>

              {/* Recursive character config */}
              {(splitterType === 'recursive' || splitterType === 'character') && (
                <>
                  <div className="form-group">
                    <label>Chunk Size (characters)</label>
                    <input 
                      type="number" 
                      className="form-control"
                      value={chunkSize}
                      onChange={(e) => setChunkSize(e.target.value)}
                    />
                  </div>
                  <div className="form-group">
                    <label>Chunk Overlap (characters)</label>
                    <input 
                      type="number" 
                      className="form-control"
                      value={chunkOverlap}
                      onChange={(e) => setChunkOverlap(e.target.value)}
                    />
                  </div>
                </>
              )}

              {/* Semantic chunking config */}
              {splitterType === 'semantic' && (
                <>
                  <div className="form-group">
                    <label>Breakpoint Threshold Type</label>
                    <select 
                      className="form-control"
                      value={breakpointThresholdType}
                      onChange={(e) => setBreakpointThresholdType(e.target.value)}
                    >
                      <option value="percentile">Percentile</option>
                      <option value="standard_deviation">Standard Deviation</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Breakpoint Threshold Percentile/Amount</label>
                    <div className="slider-container">
                      <input 
                        type="range" 
                        min="10" 
                        max="95" 
                        step="5"
                        value={breakpointThresholdAmount}
                        onChange={(e) => setBreakpointThresholdAmount(parseInt(e.target.value))}
                      />
                      <span className="slider-value">{breakpointThresholdAmount}</span>
                    </div>
                  </div>
                </>
              )}

              {/* Agentic chunking description */}
              {splitterType === 'agentic' && (
                <div style={{ fontSize: '0.8rem', background: 'rgba(139, 92, 246, 0.05)', border: '1px dashed var(--border-color)', borderRadius: '6px', padding: '0.75rem', marginBottom: '1.25rem', color: 'var(--text-secondary)' }}>
                  📌 <strong>Agentic Chunker:</strong> Uses the active LLM to evaluate text semantic bounds and split them intelligently. This is high quality but takes more processing time and api requests.
                </div>
              )}

              <button 
                className="btn btn-primary"
                style={{ marginTop: 'auto', width: '100%' }}
                onClick={handleStartIngest}
                disabled={ingestStatus.status === 'processing' || documents.length === 0}
              >
                {ingestStatus.status === 'processing' ? (
                  <>
                    <div className="loading-dots"><span></span><span></span><span></span></div>
                    Processing Pipeline...
                  </>
                ) : (
                  'Run Ingestion & Embeddings'
                )}
              </button>
            </div>

            {/* Ingestion Console Log Card (Terminal style) */}
            <div className="card terminal-card">
              <h3 className="card-title">
                Console Terminal
                <span className={`badge ${ingestStatus.status}`}>
                  Pipeline: {ingestStatus.status}
                </span>
              </h3>
              <div className="terminal-console">
                {ingestStatus.logs && ingestStatus.logs.map((log, lidx) => {
                  let logClass = 'info';
                  if (log.includes('[ERROR]')) logClass = 'error';
                  if (log.includes('[SUCCESS]') || log.includes('completed in')) logClass = 'success';
                  if (log.includes('[WARNING]')) logClass = 'warning';
                  
                  return (
                    <div key={lidx} className={`terminal-log-line ${logClass}`}>
                      {log}
                    </div>
                  );
                })}
                {(!ingestStatus.logs || ingestStatus.logs.length === 0) && (
                  <div style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
                    Console output will display here when pipeline is executed...
                  </div>
                )}
                <div ref={terminalEndRef} />
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
