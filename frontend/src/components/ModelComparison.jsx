import React from 'react';

export default function ModelComparison({ models }) {
  return (
    <div style={{ padding: '20px' }}>
      <h2>Model Comparison</h2>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th style={{ border: '1px solid #ddd', padding: '8px' }}>Model</th>
            <th style={{ border: '1px solid #ddd', padding: '8px' }}>Accuracy</th>
            <th style={{ border: '1px solid #ddd', padding: '8px' }}>Speed (ms)</th>
          </tr>
        </thead>
        <tbody>
          {models.map((model, idx) => (
            <tr key={idx}>
              <td style={{ border: '1px solid #ddd', padding: '8px' }}>{model.name}</td>
              <td style={{ border: '1px solid #ddd', padding: '8px' }}>{model.accuracy}</td>
              <td style={{ border: '1px solid #ddd', padding: '8px' }}>{model.speed}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
