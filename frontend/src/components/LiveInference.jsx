import React, { useState } from 'react';
import { api } from '../services/api';

export default function LiveInference() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    if (!file) return;
    
    setLoading(true);
    try {
      const data = await api.predict(file);
      setResult(data);
    } catch (error) {
      console.error('Prediction failed:', error);
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: '20px', border: '1px solid #ccc', borderRadius: '8px' }}>
      <h2>Live Inference</h2>
      <input 
        type="file" 
        accept="image/*"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <button onClick={handlePredict} disabled={!file || loading}>
        {loading ? 'Processing...' : 'Predict'}
      </button>
      
      {result && (
        <div style={{ marginTop: '20px' }}>
          <h3>Results:</h3>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
