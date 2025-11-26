import axios from 'axios';

const API_BASE = '/api';

export const api = {
  async predict(file) {
    const formData = new FormData();
    formData.append('file', file);
    const response = await axios.post(`${API_BASE}/inference/predict`, formData);
    return response.data;
  },

  async startTraining(config) {
    const response = await axios.post(`${API_BASE}/training/start`, config);
    return response.data;
  },

  async getTrainingStatus(jobId) {
    const response = await axios.get(`${API_BASE}/training/status/${jobId}`);
    return response.data;
  },

  async getMetrics(modelId) {
    const response = await axios.get(`${API_BASE}/metrics/model/${modelId}`);
    return response.data;
  }
};
