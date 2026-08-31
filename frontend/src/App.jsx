import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const BACKEND_URL = 'http://localhost:8000';

function App() {
  const [documents, setDocuments] = useState([]);
  const [selectedDocIds, setSelectedDocIds] = useState([]);
  const [chatHistory, setChatHistory] = useState([]);
  const [inputText, setInputText] = useState('');
  const [uploading, setUploading] = useState(false);
  const [querying, setQuerying] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);

  // Fetch indexed documents
  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/documents`);
      if (response.ok) {
        const data = await response.json();
        setDocuments(data);
      }
    } catch (error) {
      console.error("Error fetching documents:", error);
    }
  };

  useEffect(() => {
    fetchDocuments();
    // Poll documents list periodically (every 5 seconds) to show newly indexed PDFs automatically
    const interval = setInterval(fetchDocuments, 5000);
    return () => clearInterval(interval);
  }, []);

  // Scroll to bottom of chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, querying]);

  // Handle file upload
  const handleUploadFile = async (file) => {
    if (!file || file.type !== 'application/pdf') {
      alert("Please upload a valid PDF file.");
      return;
    }
    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${BACKEND_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        // Refresh document list
        fetchDocuments();
      } else {
        alert("Upload failed. Please try again.");
      }
    } catch (error) {
      console.error("Upload error:", error);
      alert("Error uploading file.");
    } finally {
      setUploading(false);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleUploadFile(e.dataTransfer.files[0]);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  // Handle document deletion
  const handleDeleteDoc = async (docId, e) => {
    e.stopPropagation();
    if (!confirm("Are you sure you want to delete this document?")) return;

    try {
      const response = await fetch(`${BACKEND_URL}/documents/${docId}`, {
        method: 'DELETE',
      });
      if (response.ok) {
        setDocuments(documents.filter(d => d.doc_id !== docId));
        setSelectedDocIds(selectedDocIds.filter(id => id !== docId));
      }
    } catch (error) {
      console.error("Error deleting document:", error);
    }
  };

  // Document filter selection toggling
  const toggleDocSelection = (docId) => {
    if (selectedDocIds.includes(docId)) {
      setSelectedDocIds(selectedDocIds.filter(id => id !== docId));
    } else {
      setSelectedDocIds([...selectedDocIds, docId]);
    }
  };

  // Send Query to RAG system
  const handleQuery = async (e) => {
    e.preventDefault();
    if (!inputText.trim()) return;

    const userMessage = { role: 'user', content: inputText.trim() };
    setChatHistory(prev => [...prev, userMessage]);
    setInputText('');
    setQuerying(true);

    try {
      const response = await fetch(`${BACKEND_URL}/query_sync`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: userMessage.content,
          top_k: 5,
          doc_ids: selectedDocIds.length > 0 ? selectedDocIds : null
        })
      });

      if (response.ok) {
        const data = await response.json();
        setChatHistory(prev => [...prev, {
          role: 'assistant',
          content: data.answer,
          citations: data.citations
        }]);
      } else {
        setChatHistory(prev => [...prev, {
          role: 'assistant',
          content: "Sorry, I encountered an error processing your query."
        }]);
      }
    } catch (error) {
      console.error("Query error:", error);
      setChatHistory(prev => [...prev, {
        role: 'assistant',
        content: "Sorry, I couldn't connect to the server."
      }]);
    } finally {
      setQuerying(false);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="sidebar-header">
          <span>📚</span> RAG Document Chat
        </div>

        <div className="sidebar-section-title">Indexed Documents</div>
        <div className="doc-list">
          {documents.length === 0 ? (
            <div className="no-docs">No PDFs indexed yet. Upload one!</div>
          ) : (
            documents.map((doc) => (
              <div 
                key={doc.doc_id} 
                className={`doc-item ${selectedDocIds.includes(doc.doc_id) ? 'selected' : ''}`}
                onClick={() => toggleDocSelection(doc.doc_id)}
              >
                <div className="doc-info">
                  <input 
                    type="checkbox" 
                    className="doc-checkbox"
                    checked={selectedDocIds.includes(doc.doc_id)}
                    onChange={() => {}} // handled by parent click
                  />
                  <span className="doc-name" title={doc.filename}>{doc.filename}</span>
                </div>
                <button 
                  className="delete-btn"
                  onClick={(e) => handleDeleteDoc(doc.doc_id, e)}
                  title="Delete Document"
                >
                  🗑️
                </button>
              </div>
            ))
          )}
        </div>

        {documents.length > 0 && (
          <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', textAlign: 'center' }}>
            {selectedDocIds.length > 0 
              ? `Searching ${selectedDocIds.length} selected document(s)` 
              : "Searching all documents"}
          </div>
        )}
      </div>

      {/* Main Content Area */}
      <div className="main-content">
        <div className="main-header">
          <h1 className="main-title">AI-Powered Document Assistant</h1>
          <p className="main-subtitle">Upload PDFs, index them into vector space, and search through them using LLaMA & Gemini.</p>
        </div>

        {/* Drag and drop upload zone */}
        <div 
          className={`upload-container ${dragActive ? 'active' : ''}`}
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          onClick={triggerFileInput}
        >
          <input 
            type="file" 
            className="file-input"
            ref={fileInputRef}
            onChange={(e) => handleUploadFile(e.target.files[0])}
            accept="application/pdf"
          />
          <div className="upload-icon">📥</div>
          {uploading ? (
            <div className="spinner-container">
              <div className="spinner"></div>
              <span>Ingesting PDF... Please wait.</span>
            </div>
          ) : (
            <>
              <div className="upload-text">Drag & drop your PDF here, or click to browse</div>
              <div className="upload-subtext">Supports normal and scanned PDF files</div>
            </>
          )}
        </div>

        {/* Chat interface */}
        <div className="chat-section">
          <div className="chat-history">
            {chatHistory.length === 0 ? (
              <div style={{ color: 'var(--color-text-muted)', textAlign: 'center', marginTop: '20%' }}>
                <div style={{ fontSize: '3rem', marginBottom: '10px' }}>💬</div>
                <p>Hello! Ask a question about your indexed documents to get started.</p>
              </div>
            ) : (
              chatHistory.map((msg, idx) => (
                <div key={idx} className={`message-bubble ${msg.role}`}>
                  <div className="avatar">
                    {msg.role === 'user' ? 'U' : 'AI'}
                  </div>
                  <div className="message-text-wrapper">
                    <div className="message-text">{msg.content}</div>
                    {msg.citations && msg.citations.length > 0 && (
                      <div className="citations-container">
                        <div className="citations-title">📎 Citations Used:</div>
                        <div className="citations-list">
                          {msg.citations.map((cit, cidx) => (
                            <div key={cidx} className="citation-card">
                              <div className="citation-header">
                                <span className="citation-source">[{cit.number}] {cit.source}</span>
                                <span className="citation-score">Relevance: {(cit.score * 100).toFixed(1)}%</span>
                              </div>
                              <div className="citation-preview">{cit.preview}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
            {querying && (
              <div className="message-bubble assistant">
                <div className="avatar">AI</div>
                <div className="spinner-container" style={{ marginLeft: '10px' }}>
                  <div className="spinner"></div>
                  <span>Thinking...</span>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          <form className="chat-form" onSubmit={handleQuery}>
            <input 
              type="text" 
              className="chat-input"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Ask a question about your indexed documents..."
              disabled={querying}
            />
            <button className="send-btn" type="submit" disabled={querying || !inputText.trim()}>
              Ask
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
