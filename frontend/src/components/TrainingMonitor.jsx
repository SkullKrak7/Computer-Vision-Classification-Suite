import React, { useState, useEffect } from 'react';
import { api } from '../services/api';

export default function TrainingMonitor() {
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null);

  const startTraining = async () => {
    const config = {
      dataset_path: 'datasets/intel_images',
      model_type: 'pytorch',
      epochs: 10,
      batch_size: 32
    };
    const result = await api.startTraining(config);
    setJobId(result.job_id);
  };

  useEffect(() => {
    if (!jobId) return;
    
    const interval = setInterval(async () => {
      const data = await api.getTrainingStatus(jobId);
      setStatus(data);
      if (data.status === 'completed') {
        clearInterval(interval);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [jobId]);

  return (
    <div style={{ padding: '20px', border: '1px solid #ccc', borderRadius: '8px' }}>
      <h2>Training Monitor</h2>
      <button onClick={startTraining}>Start Training</button>
      
      {status && (
        <div style={{ marginTop: '20px' }}>
          <p>Status: {status.status}</p>
          <p>Progress: {(status.progress * 100).toFixed(1)}%</p>
        </div>
      )}
    </div>
  );
}
