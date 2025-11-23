import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const sendMessage = async (query) => {
  try {
    const response = await apiClient.post('/chat', { query });
    return response.data;
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.detail || 'Server error occurred');
    } else if (error.request) {
      throw new Error('Unable to reach the server. Please check if the backend is running.');
    } else {
      throw new Error('An unexpected error occurred');
    }
  }
};

export const checkHealth = async () => {
  try {
    const response = await apiClient.get('/health');
    return response.data;
  } catch (error) {
    throw new Error('Health check failed');
  }
};
