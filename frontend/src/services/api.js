const API_URL = "http://localhost:8000";

export const checkHealth = async () => {
  const response = await fetch(`${API_URL}/health`);
  return response.json();
};

export const sendMessage = async (query, sessionId) => {
  const response = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ 
      query: query,
      session_id: sessionId
    }),
  });

  if (!response.ok) {
    throw new Error('Failed to send message');
  }

  return response.json();
};