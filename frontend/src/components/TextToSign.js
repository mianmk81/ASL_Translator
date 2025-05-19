import React, { useState } from 'react';
import {
  Box,
  TextField,
  Button,
  Typography,
  Card,
  CardContent,
  Grid,
  Paper,
} from '@mui/material';
import axios from 'axios';

const TextToSign = () => {
  const [inputText, setInputText] = useState('');
  const [signs, setSigns] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleGetSigns = async () => {
    if (!inputText.trim()) return;

    setLoading(true);
    try {
      // Split the input text into words
      const words = inputText.toLowerCase().trim().split(/\s+/);
      
      console.log("Sending requests for words:", words);  // Debug log
      
      // Get sign representations for each word
      const signPromises = words.map(word =>
        axios.get(`http://localhost:8000/signs/${word}`)
      );
      
      const responses = await Promise.all(signPromises);
      console.log("Received responses:", responses);  // Debug log
      
      const signData = responses.map(response => response.data).flat();
      console.log("Processed sign data:", signData);  // Debug log
      
      setSigns(signData);
    } catch (error) {
      console.error('Error fetching signs:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Typography variant="h4" gutterBottom>
        Text to Sign Language
      </Typography>

      <Card>
        <CardContent>
          <TextField
            fullWidth
            multiline
            rows={4}
            variant="outlined"
            label="Enter text to convert to signs"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            sx={{ mb: 2 }}
          />

          <Button
            variant="contained"
            onClick={handleGetSigns}
            disabled={!inputText.trim() || loading}
            sx={{ mb: 3 }}
          >
            Show Signs
          </Button>

          {loading ? (
            <Typography>Loading signs...</Typography>
          ) : signs.length > 0 ? (
            <Grid container spacing={2}>
              {signs.map((sign, index) => (
                <Grid item xs={12} sm={6} md={4} key={index}>
                  <Paper
                    sx={{
                      p: 2,
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                    }}
                  >
                    <img
                      src={sign.image_url}
                      alt={sign.word}
                      style={{
                        width: '100%',
                        maxWidth: 200,
                        height: 'auto',
                      }}
                    />
                    <Typography variant="h6" sx={{ mt: 1 }}>
                      {sign.word}
                    </Typography>
                    {sign.description && (
                      <Typography variant="body2" color="text.secondary">
                        {sign.description}
                      </Typography>
                    )}
                  </Paper>
                </Grid>
              ))}
            </Grid>
          ) : (
            <Typography color="text.secondary">
              Enter text and click "Show Signs" to see the sign language representations
            </Typography>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default TextToSign;
