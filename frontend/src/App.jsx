import React, { useState } from 'react';
import LiveInference from './components/LiveInference';
import MetricsChart from './components/MetricsChart';
import TrainingMonitor from './components/TrainingMonitor';
import ModelComparison from './components/ModelComparison';
import DatasetStats from './components/DatasetStats';

function App() {
  const [selectedModel, setSelectedModel] = useState('model1');
  
  const mockModels = [
    { name: 'PyTorch CNN', accuracy: 0.95, speed: 12 },
    { name: 'TensorFlow MobileNet', accuracy: 0.93, speed: 8 }
  ];

  const mockDataset = [
    { name: 'Class A', value: 400 },
    { name: 'Class B', value: 300 },
    { name: 'Class C', value: 200 }
  ];

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1>Computer Vision Classification Suite</h1>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginTop: '20px' }}>
        <LiveInference />
        <TrainingMonitor />
      </div>

      <div style={{ marginTop: '20px' }}>
        <MetricsChart modelId={selectedModel} />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginTop: '20px' }}>
        <ModelComparison models={mockModels} />
        <DatasetStats data={mockDataset} />
      </div>
    </div>
  );
}

export default App;
