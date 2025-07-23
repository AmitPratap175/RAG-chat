import React, { useState, useEffect, ChangeEvent, KeyboardEvent, useRef } from 'react';
import { useWebSocket } from './services/useWebSocket';
import EE from './components/easter_egg/ee';
import './App.css';

const App: React.FC = () => {
  const [messages, setMessages] = useState<{ user: string; msg: string }[]>([
    { user: 'Bot', msg: 'Welcome! How can I help you today?' }
  ]);
  const [input, setInput] = useState('');
  const [showEE, setShowEE] = useState(false);

  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const wsProtocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
  const { response, isOpen, sendMessage } = useWebSocket(`${wsProtocol}${window.location.hostname}:8000/ws`, setShowEE);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (response) {
      setMessages((prev) => [...prev, { user: 'Bot', msg: response }]);
    }
  }, [response]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => setInput(e.target.value);

  const handleSubmit = () => {
    if (input.trim()) {
      setMessages([...messages, { user: 'User', msg: input }]);
      setInput('');
      if (isOpen) sendMessage(input);
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSubmit();
    }
  };

  // PDF upload handler -- NOW in chat input form, not sidebar
  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    setUploadError(null);
    const file = event.target.files?.[0];
    if (!file) return;

    if (file.type !== "application/pdf") {
      setUploadError("Please upload a valid PDF file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setUploading(true);
      const response = await fetch('/api/upload-pdf', {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Upload failed.");

      setUploading(false);
      setMessages(prev => [...prev, { user: 'Bot', msg: `Successfully uploaded: ${file.name}` }]);
    } catch (error: any) {
      setUploadError(error.message || "Unknown error during upload");
      setUploading(false);
    }
    event.target.value = ""; // Reset file input
  };

  // Message formatting helper
  const renderBotMsg = (text: string) => {
    const lines = text.split('\n').filter(l => l.trim() !== '');
    const bulletPattern = /^\s*([*•\-])\s+/;
    const bulletLines = lines.filter(line => bulletPattern.test(line));
    if (bulletLines.length > 0 && bulletLines.length >= Math.max(2, lines.length - 1)) {
      return (
        <ul style={{ textAlign: "left", paddingLeft: "1.4em" }}>
          {lines.map((line, idx) =>
            bulletPattern.test(line) ? (
              <li key={idx}>{line.replace(bulletPattern, '')}</li>
            ) : null
          )}
        </ul>
      );
    }
    return (
      <span style={{ display: "block", textAlign: "left", whiteSpace: "pre-wrap" }}>
        {text}
      </span>
    );
  };

  return (
    <div className="bw-root">
      {/* Sidebar */}
      <aside className="bw-sidebar">
        <div className="sidebar-header">
          <img src="/normal_portrait.svg" alt="Brahmware logo" className="sidebar-logo" />
        </div>
        <nav className="sidebar-nav">
          <ul>
            <li className="selected">New Chat</li>
            <li>Support</li>
            <li>Bots</li>
            <li>More</li>
          </ul>
        </nav>
        <div className="sidebar-bottom">
          <button className="sidebar-pricing-btn">Pricing</button>
        </div>
      </aside>

      {/* Main Chat UI */}
      <main className="bw-chat-bg">
        <div className="bw-chat-header">
          <span>FicAssist</span>
        </div>
        <div className="bw-chat-messages">
          {messages.map((msg, idx) =>
            msg.user === 'User' ? (
              <div key={idx} className="chat-bubble user">
                <span className="bubble-text">{msg.msg}</span>
              </div>
            ) : (
              <div key={idx} className="chat-bubble bot">
                <span className="bubble-text">{renderBotMsg(msg.msg)}</span>
              </div>
            )
          )}
          <div ref={messagesEndRef} />
        </div>
        {/* CHAT INPUT + PDF UPLOAD BUTTON */}
        <form
          className="bw-chat-input"
          onSubmit={e => {
            e.preventDefault();
            handleSubmit();
          }}
        >
          <label htmlFor="pdf-upload" className="upload-pdf-btn" style={{ marginRight: '10px' }}>
            <span role="img" aria-label="upload" style={{ fontSize: '1.5em' }}>📎</span>
            <input
              id="pdf-upload"
              type="file"
              accept="application/pdf"
              style={{ display: 'none' }}
              onChange={handleFileChange}
              disabled={uploading}
            />
          </label>
          <input
            type="text"
            placeholder="Type your message..."
            value={input}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            aria-label="Type your message"
            autoFocus
            spellCheck={false}
          />
          <button type="submit" disabled={!input.trim()}>
            Send
          </button>
        </form>
        {/* Upload error feedback (choose your color scheme) */}
        {uploadError && (
          <div style={{ color: '#D94D4D', marginLeft: '1.7rem', marginBottom: "0.5em" }}>{uploadError}</div>
        )}
        {showEE && <EE />}
      </main>
    </div>
  );
};

export default App;
