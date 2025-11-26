import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { api } from '../services/api';

export default function MetricsChart({ modelId }) {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    if (modelId) {
      api.getMetrics(modelId).then(setMetrics);
    }
  }, [modelId]);

  if (!metrics) return <div>Loading metrics...</div>;

  const data = [
    { name: 'Accuracy', value: metrics.accuracy },
    { name: 'Precision', value: metrics.precision },
    { name: 'Recall', value: metrics.recall },
    { name: 'F1 Score', value: metrics.f1_score }
  ];

  return (
    <div>
      <h2>Model Metrics</h2>
      <BarChart width={600} height={300} data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis domain={[0, 1]} />
        <Tooltip />
        <Legend />
        <Bar dataKey="value" fill="#8884d8" />
      </BarChart>
    </div>
  );
}
