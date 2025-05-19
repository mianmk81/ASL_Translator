import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Tabs,
  Tab,
  Box,
  Container,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  List,
  ListItem,
  ListItemText,
  Divider,
  CircularProgress,
} from '@mui/material';
import {
  ContentCopy,
  VolumeUp,
  Delete,
} from '@mui/icons-material';
import SignToText from './SignToText';
import TextToSign from './TextToSign';
import DataCollection from './DataCollection';
import ModelTraining from './ModelTraining';
import {
  FuturisticCard,
  HologramBox,
  CircuitBackground,
  GlassContainer,
  NeonButton,
  DataStream,
} from './shared/StyledComponents';

const LANGUAGES = [
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Spanish' },
  { code: 'fr', name: 'French' },
  { code: 'de', name: 'German' },
  { code: 'zh', name: 'Chinese' },
];

const MainLayout = () => {
  const [currentTab, setCurrentTab] = useState(0);
  const [targetLanguage, setTargetLanguage] = useState('en');
  const [translatedText, setTranslatedText] = useState('');
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  const handleTabChange = (event, newValue) => {
    setCurrentTab(newValue);
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
  };

  const handleTextToSpeech = (text) => {
    // Text-to-speech implementation
  };

  const clearHistory = () => {
    setHistory([]);
  };

  const renderContent = () => {
    switch (currentTab) {
      case 0:
        return <SignToText />;
      case 1:
        return <TextToSign />;
      case 2:
        return <DataCollection />;
      case 3:
        return <ModelTraining />;
      default:
        return null;
    }
  };

  return (
    <CircuitBackground>
      <AppBar position="static" elevation={0}>
        <Container maxWidth="xl">
          <Toolbar disableGutters>
            <Typography
              variant="h4"
              sx={{
                fontFamily: 'Orbitron',
                background: 'linear-gradient(45deg, #00f2ff 30%, #ff0099 90%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                letterSpacing: '0.2rem',
                fontWeight: 700,
              }}
            >
              ASL Translator
            </Typography>
          </Toolbar>
        </Container>
      </AppBar>

      <Container maxWidth="xl" sx={{ mt: 4, pb: 4 }}>
        <GlassContainer sx={{ mb: 4 }}>
          <Tabs
            value={currentTab}
            onChange={handleTabChange}
            variant="fullWidth"
            textColor="primary"
            sx={{
              '& .MuiTab-root': {
                color: 'text.secondary',
                fontFamily: 'Orbitron',
                letterSpacing: '0.1rem',
                '&.Mui-selected': {
                  color: '#00f2ff',
                },
              },
              '& .MuiTabs-indicator': {
                background: 'linear-gradient(45deg, #00f2ff 30%, #ff0099 90%)',
                height: 3,
              },
            }}
          >
            <Tab label="Sign to Text" />
            <Tab label="Text to Sign" />
            <Tab label="Data Collection" />
            <Tab label="Model Training" />
          </Tabs>
        </GlassContainer>

        <FuturisticCard>
          <Box sx={{ minHeight: '70vh' }}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                {renderContent()}
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Box sx={{ p: 2 }}>
                  <GlassContainer>
                    <Box sx={{ mb: 2 }}>
                      <FormControl fullWidth size="small">
                        <InputLabel sx={{ color: '#8892b0' }}>Target Language</InputLabel>
                        <Select
                          value={targetLanguage}
                          onChange={(e) => setTargetLanguage(e.target.value)}
                          label="Target Language"
                          sx={{
                            '& .MuiOutlinedInput-notchedOutline': {
                              borderColor: 'rgba(0, 242, 255, 0.2)',
                            },
                            '&:hover .MuiOutlinedInput-notchedOutline': {
                              borderColor: 'rgba(0, 242, 255, 0.4)',
                            },
                            '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                              borderColor: '#00f2ff',
                            },
                          }}
                        >
                          {LANGUAGES.map(lang => (
                            <MenuItem key={lang.code} value={lang.code}>
                              {lang.name}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Box>

                    <Typography variant="h6" gutterBottom sx={{ color: '#e6f1ff' }}>
                      Translation
                    </Typography>
                    
                    {loading ? (
                      <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
                        <CircularProgress sx={{ color: '#00f2ff' }} />
                      </Box>
                    ) : translatedText ? (
                      <Box sx={{ mb: 2 }}>
                        <DataStream sx={{ p: 2 }}>
                          <Typography sx={{ color: '#00f2ff' }}>{translatedText}</Typography>
                        </DataStream>
                        <Box sx={{ mt: 1, display: 'flex', gap: 1 }}>
                          <NeonButton
                            onClick={() => copyToClipboard(translatedText)}
                            startIcon={<ContentCopy />}
                            size="small"
                          >
                            Copy
                          </NeonButton>
                          <NeonButton
                            onClick={() => handleTextToSpeech(translatedText)}
                            startIcon={<VolumeUp />}
                            size="small"
                          >
                            Speak
                          </NeonButton>
                        </Box>
                      </Box>
                    ) : (
                      <Typography sx={{ color: '#8892b0' }}>
                        Translation will appear here
                      </Typography>
                    )}

                    <Divider sx={{ my: 2, borderColor: 'rgba(0, 242, 255, 0.1)' }} />

                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                      <Typography variant="h6" sx={{ color: '#e6f1ff' }}>History</Typography>
                      <IconButton 
                        size="small" 
                        onClick={clearHistory} 
                        disabled={!history.length}
                        sx={{ color: '#ff0099' }}
                      >
                        <Delete />
                      </IconButton>
                    </Box>

                    {history.length > 0 ? (
                      <List>
                        {history.map((entry, index) => (
                          <ListItem key={index} sx={{ 
                            borderBottom: '1px solid rgba(0, 242, 255, 0.1)',
                            '&:last-child': { borderBottom: 'none' },
                          }}>
                            <ListItemText
                              primary={
                                <Typography sx={{ color: '#e6f1ff' }}>{entry.text}</Typography>
                              }
                              secondary={
                                <Typography sx={{ color: '#8892b0', fontSize: '0.8rem' }}>
                                  {new Date(entry.timestamp).toLocaleString()}
                                </Typography>
                              }
                            />
                          </ListItem>
                        ))}
                      </List>
                    ) : (
                      <Typography sx={{ color: '#8892b0' }}>
                        No history yet
                      </Typography>
                    )}
                  </GlassContainer>
                </Box>
              </Grid>
            </Grid>
          </Box>
        </FuturisticCard>
      </Container>
    </CircuitBackground>
  );
};

export default MainLayout;
