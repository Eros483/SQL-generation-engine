import React, { useState, useRef, useEffect } from 'react';
import { sendMessage, checkHealth } from '../services/api';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import '../styles/ChatInterface.css';

function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('checking');
  const [sessionId, setSessionId] = useState('');
  const messagesEndRef = useRef(null);

  useEffect(() => {
    checkBackendHealth();
    let storedSessionId = localStorage.getItem("chat_session_id");
    if (!storedSessionId) {
      storedSessionId = crypto.randomUUID();
      localStorage.setItem("chat_session_id", storedSessionId);
    }
    setSessionId(storedSessionId);
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const checkBackendHealth = async () => {
    try {
      const health = await checkHealth();
      setConnectionStatus(health.status === 'healthy' ? 'connected' : 'error');
    } catch (error) {
      setConnectionStatus('error');
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async (text) => {
    if (!text.trim()) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: text,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await sendMessage(text, sessionId);

      const aiMessage = {
        id: Date.now() + 1,
        type: 'ai',
        content: response.response,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        type: 'error',
        content: error.message,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearChat = () => {
    setMessages([]);
    const newSessionId = crypto.randomUUID();
    setSessionId(newSessionId);
    localStorage.setItem("chat_session_id", newSessionId);
  };

  return (
    <div className="chat-interface">
      <header className="navbar">
        <div className="navbar-container">
          <div className="navbar-logo">
            <img src="/foresighthealth_logo.jpeg" alt="Foresight Health Logo" className="logo-icon" />
            <span className="logo-text">Caliper</span>
          </div>

          <nav className="navbar-links">
            <a href="#modules" className="nav-link">Modules</a>
            <a href="#clients" className="nav-link">Clients</a>
            <a href="#about" className="nav-link">About</a>
            <a href="#resources" className="nav-link">Resources</a>
          </nav>

          <div className="navbar-cta">
            <button className="cta-button">
              <span>Book a demo</span>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <rect x="3" y="4" width="18" height="18" rx="2"/>
                <path d="M16 2v4M8 2v4M3 10h18"/>
              </svg>
            </button>
          </div>
        </div>
      </header>

      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="empty-container">
              <div className="empty-icon">
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                </svg>
              </div>
              <h2>AI-Powered Database Insights</h2>
              <p className="empty-subtitle">Ask natural language questions about your healthcare data</p>

              <div className="example-queries">
                <p className="examples-title">Example queries:</p>
                <div className="example-grid">
                  <div className="example-item">"Show all patients with housing assistance"</div>
                  <div className="example-item">"How many interventions were provided last month?"</div>
                  <div className="example-item">"List high-risk patients needing follow-up"</div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <>
            <div className="messages-container">
              {messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
              ))}
              {isLoading && (
                <div className="loading-message">
                  <div className="loading-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </>
        )}
      </div>

      <footer className="chat-footer">
        <div className="footer-status">
          <div className={`status-badge ${connectionStatus}`}>
            <span className="status-dot"></span>
            <span className="status-text">
              {connectionStatus === 'connected' && 'Connected'}
              {connectionStatus === 'error' && 'Disconnected'}
              {connectionStatus === 'checking' && 'Checking...'}
            </span>
          </div>
          {messages.length > 0 && (
            <button onClick={handleClearChat} className="footer-action">
              Clear conversation
            </button>
          )}
        </div>
        <ChatInput onSendMessage={handleSendMessage} disabled={isLoading} />
      </footer>
    </div>
  );
}

export default ChatInterface;
