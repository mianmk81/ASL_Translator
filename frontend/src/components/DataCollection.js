import React, { useState, useRef } from 'react';
import {
  Box,
  Typography,
  Grid,
  TextField,
  IconButton,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
} from '@mui/material';
import {
  PhotoCamera,
  Delete,
  CloudUpload,
  Check,
} from '@mui/icons-material';
import {
  FuturisticCard,
  GlassContainer,
  NeonButton,
  DataStream,
  HologramBox,
} from './shared/StyledComponents';

const DataCollection = () => {
  const videoRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [captures, setCaptures] = useState([]);
  const [currentSign, setCurrentSign] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  // Start camera
  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: 640,
          height: 480,
          facingMode: 'user'
        } 
      });
      setStream(mediaStream);
      videoRef.current.srcObject = mediaStream;
    } catch (error) {
      console.error('Error accessing camera:', error);
    }
  };

  // Stop camera
  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
      videoRef.current.srcObject = null;
    }
  };

  // Capture frame
  const captureFrame = () => {
    if (!currentSign.trim()) {
      alert('Please enter the sign label first');
      return;
    }

    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    canvas.getContext('2d').drawImage(videoRef.current, 0, 0);
    
    const capture = {
      id: Date.now(),
      image: canvas.toDataURL('image/jpeg'),
      label: currentSign,
      timestamp: new Date().toISOString(),
    };

    setCaptures(prev => [...prev, capture]);
  };

  // Upload captures
  const uploadCaptures = async () => {
    if (captures.length === 0) return;

    setUploadProgress(0);
    const total = captures.length;

    for (let i = 0; i < captures.length; i++) {
      try {
        const formData = new FormData();
        formData.append('image', dataURLtoBlob(captures[i].image));
        formData.append('label', captures[i].label);
        
        await fetch('http://localhost:8000/upload-training-data', {
          method: 'POST',
          body: formData,
        });

        setUploadProgress(((i + 1) / total) * 100);
      } catch (error) {
        console.error('Error uploading capture:', error);
      }
    }

    // Clear captures after successful upload
    setCaptures([]);
    setUploadProgress(0);
  };

  // Convert dataURL to Blob
  const dataURLtoBlob = (dataURL) => {
    const arr = dataURL.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    return new Blob([u8arr], { type: mime });
  };

  // Delete capture
  const deleteCapture = (id) => {
    setCaptures(prev => prev.filter(capture => capture.id !== id));
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
        Data Collection Interface
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <GlassContainer>
            <Box sx={{ position: 'relative', mb: 3 }}>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                style={{
                  width: '100%',
                  borderRadius: '8px',
                  border: '1px solid rgba(0, 242, 255, 0.2)',
                }}
              />
              
              {!stream && (
                <Box
                  sx={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: 'rgba(0, 0, 0, 0.7)',
                    borderRadius: '8px',
                  }}
                >
                  <NeonButton onClick={startCamera}>
                    Start Camera
                  </NeonButton>
                </Box>
              )}
            </Box>

            <Box sx={{ mb: 3 }}>
              <TextField
                fullWidth
                variant="outlined"
                label="Sign Label"
                value={currentSign}
                onChange={(e) => setCurrentSign(e.target.value)}
                sx={{
                  '& .MuiOutlinedInput-root': {
                    color: '#e6f1ff',
                    '& fieldset': {
                      borderColor: 'rgba(0, 242, 255, 0.2)',
                    },
                    '&:hover fieldset': {
                      borderColor: 'rgba(0, 242, 255, 0.4)',
                    },
                    '&.Mui-focused fieldset': {
                      borderColor: '#00f2ff',
                    },
                  },
                  '& .MuiInputLabel-root': {
                    color: '#8892b0',
                  },
                }}
              />
            </Box>

            <Box sx={{ display: 'flex', gap: 2 }}>
              <NeonButton
                onClick={captureFrame}
                disabled={!stream}
                startIcon={<PhotoCamera />}
              >
                Capture Frame
              </NeonButton>
              <NeonButton
                onClick={stopCamera}
                disabled={!stream}
                color="error"
              >
                Stop Camera
              </NeonButton>
            </Box>
          </GlassContainer>
        </Grid>

        <Grid item xs={12} md={6}>
          <HologramBox>
            <Typography variant="h6" sx={{ color: '#e6f1ff', mb: 2 }}>
              Captured Signs
            </Typography>

            {captures.length > 0 ? (
              <>
                <List>
                  {captures.map((capture) => (
                    <ListItem
                      key={capture.id}
                      sx={{
                        border: '1px solid rgba(0, 242, 255, 0.2)',
                        borderRadius: '4px',
                        mb: 1,
                      }}
                    >
                      <img
                        src={capture.image}
                        alt={capture.label}
                        style={{
                          width: '60px',
                          height: '60px',
                          objectFit: 'cover',
                          borderRadius: '4px',
                          marginRight: '16px',
                        }}
                      />
                      <ListItemText
                        primary={
                          <Typography sx={{ color: '#e6f1ff' }}>
                            {capture.label}
                          </Typography>
                        }
                        secondary={
                          <Typography sx={{ color: '#8892b0', fontSize: '0.8rem' }}>
                            {new Date(capture.timestamp).toLocaleString()}
                          </Typography>
                        }
                      />
                      <ListItemSecondaryAction>
                        <IconButton
                          edge="end"
                          onClick={() => deleteCapture(capture.id)}
                          sx={{ color: '#ff0099' }}
                        >
                          <Delete />
                        </IconButton>
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))}
                </List>

                <Box sx={{ mt: 2 }}>
                  <NeonButton
                    onClick={uploadCaptures}
                    startIcon={<CloudUpload />}
                    fullWidth
                  >
                    Upload Captures
                  </NeonButton>
                </Box>

                {uploadProgress > 0 && (
                  <DataStream
                    sx={{
                      mt: 2,
                      height: '4px',
                      '&::before': {
                        width: `${uploadProgress}%`,
                      },
                    }}
                  />
                )}
              </>
            ) : (
              <Typography sx={{ color: '#8892b0', textAlign: 'center' }}>
                No captures yet. Start by capturing some sign images!
              </Typography>
            )}
          </HologramBox>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DataCollection;
