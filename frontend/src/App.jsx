import React, { useState, useEffect } from 'react';
import LiveInference from './components/LiveInference';
import MetricsChart from './components/MetricsChart';
import TrainingMonitor from './components/TrainingMonitor';
import ModelComparison from './components/ModelComparison';
import DatasetStats from './components/DatasetStats';
import { api } from './services/api';

function App() {
  const [selectedModel, setSelectedModel] = useState('pytorch_cnn');
  const [allMetrics, setAllMetrics] = useState({});
  
  const models = [
    { id: 'knn', name: 'KNN Baseline' },
    { id: 'svm', name: 'SVM Baseline' },
    { id: 'pytorch_cnn', name: 'PyTorch CNN' },
    { id: 'tensorflow_mobilenet', name: 'TensorFlow MobileNetV2' }
  ];

  useEffect(() => {
    models.forEach(model => {
      api.getMetrics(model.id)
        .then(metrics => {
          setAllMetrics(prev => ({ ...prev, [model.id]: metrics }));
        })
        .catch(err => console.error(`Failed to load ${model.id}:`, err));
    });
  }, []);

  const comparisonData = models
    .filter(m => allMetrics[m.id])
    .map(m => ({
      name: m.name,
      accuracy: allMetrics[m.id].accuracy,
      f1_score: allMetrics[m.id].f1_score
    }));

  const mockDataset = [
    { name: 'Buildings', value: 2191 },
    { name: 'Forest', value: 2271 },
    { name: 'Glacier', value: 2404 },
    { name: 'Mountain', value: 2512 },
    { name: 'Sea', value: 2274 },
    { name: 'Street', value: 2382 }
  ];

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif', backgroundColor: '#f5f5f5', minHeight: '100vh' }}>
      <h1 style={{ textAlign: 'center', color: '#333' }}>Computer Vision Classification Suite</h1>
      
      <div style={{ marginTop: '20px', marginBottom: '20px', textAlign: 'center' }}>
        <label style={{ marginRight: '10px', fontWeight: 'bold' }}>Select Model: </label>
        <select 
          value={selectedModel} 
          onChange={(e) => setSelectedModel(e.target.value)}
          style={{ padding: '8px', fontSize: '14px', borderRadius: '4px' }}
        >
          {models.map(m => (
            <option key={m.id} value={m.id}>{m.name}</option>
          ))}
        </select>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginTop: '20px' }}>
        <LiveInference />
        <TrainingMonitor />
      </div>

      <div style={{ marginTop: '20px' }}>
        <MetricsChart modelId={selectedModel} />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginTop: '20px' }}>
        <ModelComparison models={comparisonData} />
        <DatasetStats data={mockDataset} />
      </div>
    </div>
  );
}

export default App;
