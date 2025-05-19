import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  LinearProgress,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import {
  FuturisticCard,
  GlassContainer,
  NeonButton,
  DataStream,
  HologramBox,
} from './shared/StyledComponents';

const ModelTraining = () => {
  const [trainingStatus, setTrainingStatus] = useState('idle'); // idle, training, completed, error
  const [progress, setProgress] = useState(0);
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(32);
  const [metrics, setMetrics] = useState({
    accuracy: [],
    loss: [],
  });
  const [modelConfig, setModelConfig] = useState({
    learningRate: 0.001,
    architecture: 'mobilenet',
    augmentation: true,
  });

  const startTraining = async () => {
    try {
      setTrainingStatus('training');
      setProgress(0);
      
      // Connect to backend websocket for real-time updates
      const ws = new WebSocket('ws://localhost:8000/ws/training');
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'progress') {
          setProgress(data.value);
          setMetrics(prevMetrics => ({
            accuracy: [...prevMetrics.accuracy, data.accuracy],
            loss: [...prevMetrics.loss, data.loss],
          }));
        } else if (data.type === 'completed') {
          setTrainingStatus('completed');
          ws.close();
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setTrainingStatus('error');
      };

      // Initialize training on backend
      const response = await fetch('http://localhost:8000/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          epochs,
          batch_size: batchSize,
          learning_rate: modelConfig.learningRate,
          architecture: modelConfig.architecture,
          use_augmentation: modelConfig.augmentation,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to start training');
      }
    } catch (error) {
      console.error('Training error:', error);
      setTrainingStatus('error');
    }
  };

  const getStatusColor = () => {
    switch (trainingStatus) {
      case 'training':
        return '#00f2ff';
      case 'completed':
        return '#00ff00';
      case 'error':
        return '#ff0099';
      default:
        return '#8892b0';
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography 
        variant="h4" 
        gutterBottom
        sx={{
          fontFamily: 'Orbitron',
          color: '#e6f1ff',
          textShadow: '0 0 10px rgba(0, 242, 255, 0.5)',
        }}
      >
        Model Training Interface
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <GlassContainer>
            <Typography variant="h6" sx={{ color: '#e6f1ff', mb: 2 }}>
              Training Configuration
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <FormControl fullWidth size="small">
                  <InputLabel sx={{ color: '#8892b0' }}>Architecture</InputLabel>
                  <Select
                    value={modelConfig.architecture}
                    onChange={(e) => setModelConfig({ ...modelConfig, architecture: e.target.value })}
                    sx={{
                      color: '#e6f1ff',
                      '& .MuiOutlinedInput-notchedOutline': {
                        borderColor: 'rgba(0, 242, 255, 0.2)',
                      },
                    }}
                  >
                    <MenuItem value="mobilenet">MobileNet V2</MenuItem>
                    <MenuItem value="resnet">ResNet50</MenuItem>
                    <MenuItem value="efficientnet">EfficientNet</MenuItem>
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={6}>
                <FormControl fullWidth size="small">
                  <InputLabel sx={{ color: '#8892b0' }}>Epochs</InputLabel>
                  <Select
                    value={epochs}
                    onChange={(e) => setEpochs(e.target.value)}
                    sx={{
                      color: '#e6f1ff',
                      '& .MuiOutlinedInput-notchedOutline': {
                        borderColor: 'rgba(0, 242, 255, 0.2)',
                      },
                    }}
                  >
                    {[5, 10, 15, 20, 25, 30].map((value) => (
                      <MenuItem key={value} value={value}>{value}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={6}>
                <FormControl fullWidth size="small">
                  <InputLabel sx={{ color: '#8892b0' }}>Batch Size</InputLabel>
                  <Select
                    value={batchSize}
                    onChange={(e) => setBatchSize(e.target.value)}
                    sx={{
                      color: '#e6f1ff',
                      '& .MuiOutlinedInput-notchedOutline': {
                        borderColor: 'rgba(0, 242, 255, 0.2)',
                      },
                    }}
                  >
                    {[16, 32, 64, 128].map((value) => (
                      <MenuItem key={value} value={value}>{value}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
            </Grid>

            <Box sx={{ mt: 3 }}>
              <NeonButton
                onClick={startTraining}
                disabled={trainingStatus === 'training'}
                fullWidth
              >
                {trainingStatus === 'training' ? 'Training in Progress' : 'Start Training'}
              </NeonButton>
            </Box>
          </GlassContainer>
        </Grid>

        <Grid item xs={12} md={6}>
          <HologramBox>
            <Typography variant="h6" sx={{ color: '#e6f1ff', mb: 2 }}>
              Training Status
            </Typography>
            
            <Box sx={{ mb: 2 }}>
              <Typography sx={{ color: getStatusColor(), mb: 1 }}>
                Status: {trainingStatus.charAt(0).toUpperCase() + trainingStatus.slice(1)}
              </Typography>
              <LinearProgress
                variant="determinate"
                value={progress}
                sx={{
                  height: 8,
                  borderRadius: 4,
                  backgroundColor: 'rgba(0, 242, 255, 0.1)',
                  '& .MuiLinearProgress-bar': {
                    backgroundColor: getStatusColor(),
                    borderRadius: 4,
                  },
                }}
              />
            </Box>

            <DataStream>
              <Typography sx={{ color: '#00f2ff', fontFamily: 'monospace' }}>
                Epoch: {Math.floor(progress / (100 / epochs))} / {epochs}
              </Typography>
              <Typography sx={{ color: '#00f2ff', fontFamily: 'monospace' }}>
                Accuracy: {metrics.accuracy[metrics.accuracy.length - 1]?.toFixed(4) || 0}
              </Typography>
              <Typography sx={{ color: '#00f2ff', fontFamily: 'monospace' }}>
                Loss: {metrics.loss[metrics.loss.length - 1]?.toFixed(4) || 0}
              </Typography>
            </DataStream>
          </HologramBox>

          <Box sx={{ mt: 3, height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={metrics.accuracy.map((acc, index) => ({
                  epoch: index + 1,
                  accuracy: acc,
                  loss: metrics.loss[index],
                }))}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0, 242, 255, 0.1)" />
                <XAxis dataKey="epoch" stroke="#e6f1ff" />
                <YAxis stroke="#e6f1ff" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    border: '1px solid #00f2ff',
                    borderRadius: '4px',
                  }}
                />
                <Legend />
                <Bar dataKey="accuracy" fill="#00f2ff" />
                <Bar dataKey="loss" fill="#ff0099" />
              </BarChart>
            </ResponsiveContainer>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ModelTraining;
