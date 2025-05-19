import React, { useRef, useEffect, useState, useCallback } from 'react';
import Webcam from 'react-webcam';
import { Box, Button, CircularProgress, Typography, Alert, Fade, Paper } from '@mui/material';
import { Videocam, VideocamOff } from '@mui/icons-material';

const SignToText = ({ onTranslation, targetLanguage }) => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const lastFrameTimeRef = useRef(0);
  const frameIntervalRef = useRef(1000 / 20); // Target 20 FPS
  const reconnectTimeoutRef = useRef(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isInitializing, setIsInitializing] = useState(true);
  const [error, setError] = useState(null);
  const [hasCamera, setHasCamera] = useState(true);

  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    // Close existing connection if any
    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch (err) {
        console.error('Error closing existing connection:', err);
      }
      wsRef.current = null;
    }

    try {
      // Get the WebSocket URL from environment or use default
      // For local development: ws://localhost:8080/ws/webcam
      // For EC2 deployment: ws://your-ec2-public-ip:8080/ws/webcam
      const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8080/ws/webcam';
      const ws = new WebSocket(wsUrl);
      
      // Set a connection timeout
      const connectionTimeout = setTimeout(() => {
        if (ws.readyState !== WebSocket.OPEN) {
          ws.close();
          setError('Connection timeout. Server might be down.');
          setIsInitializing(false);
        }
      }, 5000);

      ws.onopen = () => {
        clearTimeout(connectionTimeout);
        console.log('WebSocket connection established');
        setIsInitializing(false);
        setError(null);
        wsRef.current = ws;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.error) {
            console.error('WebSocket error:', data.error);
            setError(data.error);
            return;
          }

          if (!canvasRef.current) {
            console.warn('Canvas not ready');
            return;
          }

          // Draw landmarks on canvas
          const canvas = canvasRef.current;
          const ctx = canvas.getContext('2d');
          
          // Clear previous drawings
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          if (data.frame) {
            // Create an image from the base64 frame
            const img = new Image();
            img.onload = () => {
              if (canvasRef.current) {
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
              }
            };
            img.src = 'data:image/jpeg;base64,' + data.frame;
          }

          // Handle detected signs
          if (data.detected_signs && data.detected_signs.length > 0) {
            onTranslation(prev => (prev ? prev + ' ' + data.detected_signs[0] : data.detected_signs[0]));
          }
        } catch (err) {
          console.error('Error processing message:', err);
        }
      };

      ws.onerror = (error) => {
        clearTimeout(connectionTimeout);
        console.error('WebSocket error:', error);
        setError('Failed to connect to the server. Make sure the server is running.');
        setIsInitializing(false);
        wsRef.current = null;
      };

      ws.onclose = (event) => {
        clearTimeout(connectionTimeout);
        console.log('WebSocket connection closed:', event.code, event.reason);
        setIsRecording(false);
        wsRef.current = null;

        // Try to reconnect after 2 seconds if not a normal closure
        if (event.code !== 1000) {
          if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
          }
          reconnectTimeoutRef.current = setTimeout(() => {
            console.log('Attempting to reconnect...');
            connectWebSocket();
          }, 2000);
        }
      };

    } catch (err) {
      console.error('Error initializing WebSocket:', err);
      setError('Failed to initialize. Please make sure the server is running.');
      setIsInitializing(false);
    }
  }, [onTranslation]);

  useEffect(() => {
    const initializeWebSocket = async () => {
      try {
        setIsInitializing(true);
        setError(null);

        // Check if camera is available
        const devices = await navigator.mediaDevices.enumerateDevices();
        const hasVideoDevice = devices.some(device => device.kind === 'videoinput');
        setHasCamera(hasVideoDevice);
        
        if (!hasVideoDevice) {
          setError('No camera detected. Please connect a camera and refresh the page.');
          return;
        }

        connectWebSocket();
      } catch (err) {
        console.error('Error initializing:', err);
        setError('Failed to initialize. Please refresh the page and try again.');
      }
    };

    initializeWebSocket();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connectWebSocket]);

  useEffect(() => {
    let frameId;

    const sendFrame = async () => {
      if (!wsRef.current || !webcamRef.current?.video || !isRecording) return;

      try {
        const currentTime = performance.now();
        // Limit frame rate to 20 FPS for better stability
        if (currentTime - lastFrameTimeRef.current < frameIntervalRef.current) {
          if (isRecording) {
            frameId = requestAnimationFrame(sendFrame);
          }
          return;
        }
        lastFrameTimeRef.current = currentTime;

        const video = webcamRef.current.video;
        const canvas = document.createElement('canvas');
        // Scale down the video for better performance
        const scale = 0.75;
        canvas.width = video.videoWidth * scale;
        canvas.height = video.videoHeight * scale;
        const ctx = canvas.getContext('2d');
        
        // Mirror the video horizontally
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.setTransform(1, 0, 0, 1, 0, 0); // Reset transform
        
        // Get the frame as JPEG base64 with reduced quality
        const frame = canvas.toDataURL('image/jpeg', 0.6).split(',')[1];
        
        // Only send if WebSocket is ready and not buffering too much
        if (wsRef.current.readyState === WebSocket.OPEN && wsRef.current.bufferedAmount === 0) {
          wsRef.current.send(JSON.stringify({ frame }));
        }
        
        if (isRecording) {
          frameId = requestAnimationFrame(sendFrame);
        }
      } catch (err) {
        console.error('Error sending frame:', err);
        setError('Error processing video feed. Please try again.');
        setIsRecording(false);
      }
    };

    if (isRecording) {
      sendFrame();
    }

    return () => {
      if (frameId) {
        cancelAnimationFrame(frameId);
      }
    };
  }, [isRecording]);

  if (isInitializing) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (!hasCamera) {
    return (
      <Box sx={{ p: 2 }}>
        <Alert severity="error">
          No camera detected. Please connect a camera and refresh the page.
        </Alert>
      </Box>
    );
  }

  return (
    <Box>
      <Paper
        elevation={2}
        sx={{
          position: 'relative',
          width: '100%',
          paddingTop: '75%',
          backgroundColor: '#000',
          overflow: 'hidden',
        }}
      >
        <Fade in={true}>
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
            }}
          >
            <Webcam
              ref={webcamRef}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                objectFit: 'cover',
              }}
              mirrored={true}
            />
            <canvas
              ref={canvasRef}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
              }}
              width={640}
              height={480}
            />
          </Box>
        </Fade>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      )}

      <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center', gap: 2 }}>
        <Button
          variant="contained"
          color={isRecording ? 'secondary' : 'primary'}
          onClick={() => setIsRecording(!isRecording)}
          startIcon={isRecording ? <VideocamOff /> : <Videocam />}
          disabled={!!error}
        >
          {isRecording ? 'Stop Recording' : 'Start Recording'}
        </Button>
      </Box>

      <Typography
        variant="caption"
        color="text.secondary"
        sx={{ mt: 2, display: 'block', textAlign: 'center' }}
      >
        Position your hands clearly in front of the camera and make signs
      </Typography>
    </Box>
  );
};

export default SignToText;
