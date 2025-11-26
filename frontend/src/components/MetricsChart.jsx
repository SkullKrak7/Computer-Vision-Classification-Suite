import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { api } from '../services/api';

export default function MetricsChart({ modelId }) {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (modelId) {
      setLoading(true);
      setError(null);
      api.getMetrics(modelId)
        .then(data => {
          setMetrics(data);
          setLoading(false);
        })
        .catch(err => {
          setError(err.message);
          setLoading(false);
        });
    }
  }, [modelId]);

  if (loading) return <div style={{ padding: '20px' }}>Loading metrics...</div>;
  if (error) return <div style={{ padding: '20px', color: 'red' }}>Error: {error}</div>;
  if (!metrics) return <div style={{ padding: '20px' }}>No metrics available</div>;

  const data = [
    { name: 'Accuracy', value: metrics.accuracy * 100 },
    { name: 'Precision', value: metrics.precision * 100 },
    { name: 'Recall', value: metrics.recall * 100 },
    { name: 'F1 Score', value: metrics.f1_score * 100 }
  ];

  return (
    <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
      <h2>Model Metrics - {modelId.replace(/_/g, ' ').toUpperCase()}</h2>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis domain={[0, 100]} label={{ value: 'Percentage (%)', angle: -90, position: 'insideLeft' }} />
          <Tooltip formatter={(value) => `${value.toFixed(2)}%`} />
          <Legend />
          <Bar dataKey="value" fill="#8884d8" name="Score (%)" />
        </BarChart>
      </ResponsiveContainer>
      <div style={{ marginTop: '20px', display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '10px' }}>
        <div style={{ textAlign: 'center', padding: '10px', backgroundColor: '#f0f0f0', borderRadius: '4px' }}>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#8884d8' }}>{(metrics.accuracy * 100).toFixed(2)}%</div>
          <div style={{ fontSize: '12px', color: '#666' }}>Accuracy</div>
        </div>
        <div style={{ textAlign: 'center', padding: '10px', backgroundColor: '#f0f0f0', borderRadius: '4px' }}>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#82ca9d' }}>{(metrics.precision * 100).toFixed(2)}%</div>
          <div style={{ fontSize: '12px', color: '#666' }}>Precision</div>
        </div>
        <div style={{ textAlign: 'center', padding: '10px', backgroundColor: '#f0f0f0', borderRadius: '4px' }}>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#ffc658' }}>{(metrics.recall * 100).toFixed(2)}%</div>
          <div style={{ fontSize: '12px', color: '#666' }}>Recall</div>
        </div>
        <div style={{ textAlign: 'center', padding: '10px', backgroundColor: '#f0f0f0', borderRadius: '4px' }}>
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#ff7c7c' }}>{(metrics.f1_score * 100).toFixed(2)}%</div>
          <div style={{ fontSize: '12px', color: '#666' }}>F1 Score</div>
        </div>
      </div>
    </div>
  );
}
