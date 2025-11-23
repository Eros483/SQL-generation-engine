import React from 'react';
import '../styles/ChatMessage.css';

function ChatMessage({ message }) {
  const formatContent = (content) => {
    const lines = content.split('\n');
    return lines.map((line, index) => (
      <React.Fragment key={index}>
        {line}
        {index < lines.length - 1 && <br />}
      </React.Fragment>
    ));
  };

  return (
    <div className={`chat-message ${message.type}`}>
      <div className="message-avatar">
        {message.type === 'user' ? '👤' : message.type === 'error' ? '⚠️' : '🤖'}
      </div>
      <div className="message-content">
        <div className="message-text">
          {formatContent(message.content)}
        </div>
        <div className="message-timestamp">
          {message.timestamp.toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
}

export default ChatMessage;
