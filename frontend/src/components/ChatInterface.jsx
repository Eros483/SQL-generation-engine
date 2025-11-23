import React, { useState, useRef, useEffect } from 'react';
import { sendMessage, checkHealth } from '../services/api';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import '../styles/ChatInterface.css';

function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('checking');
  const messagesEndRef = useRef(null);

  useEffect(() => {
    checkBackendHealth();
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
      const response = await sendMessage(text);

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
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <div className="header-content">
          <h1>Caliper SQL Assistant</h1>
          <p className="subtitle">Ask questions about your database in natural language</p>
        </div>
        <div className="header-actions">
          <div className={`status-indicator ${connectionStatus}`}>
            <span className="status-dot"></span>
            {connectionStatus === 'connected' && 'Connected'}
            {connectionStatus === 'error' && 'Disconnected'}
            {connectionStatus === 'checking' && 'Checking...'}
          </div>
          {messages.length > 0 && (
            <button onClick={handleClearChat} className="clear-button">
              Clear Chat
            </button>
          )}
        </div>
      </div>

      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">💬</div>
            <h2>Start a Conversation</h2>
            <p>Ask me anything about your database</p>
            <div className="example-queries">
              <p className="examples-title">Try asking:</p>
              <div className="example-item">"Show me all patients"</div>
              <div className="example-item">"How many housing assistance interventions were provided?"</div>
              <div className="example-item">"List patients who haven't received any interventions"</div>
            </div>
          </div>
        ) : (
          <>
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
          </>
        )}
      </div>

      <ChatInput onSendMessage={handleSendMessage} disabled={isLoading} />
    </div>
  );
}

export default ChatInterface;
