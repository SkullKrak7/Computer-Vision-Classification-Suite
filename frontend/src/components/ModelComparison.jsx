import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

export default function ModelComparison({ models }) {
  if (!models || models.length === 0) {
    return <div style={{ padding: '20px' }}><h2>Model Comparison</h2><p>Loading...</p></div>;
  }

  return (
    <div style={{ padding: '20px', backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
      <h2>Model Comparison</h2>
      <BarChart width={500} height={300} data={models}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" angle={-15} textAnchor="end" height={80} />
        <YAxis domain={[0, 1]} />
        <Tooltip formatter={(value) => (value * 100).toFixed(2) + '%'} />
        <Legend />
        <Bar dataKey="accuracy" fill="#8884d8" name="Accuracy" />
        <Bar dataKey="f1_score" fill="#82ca9d" name="F1 Score" />
      </BarChart>
    </div>
  );
}
