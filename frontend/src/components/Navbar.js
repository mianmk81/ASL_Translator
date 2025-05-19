import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';
import SignLanguageIcon from '@mui/icons-material/SignLanguage';

const Navbar = () => {
  return (
    <AppBar position="static">
      <Toolbar>
        <SignLanguageIcon sx={{ mr: 2 }} />
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Sign Language Translator
        </Typography>
        <Box>
          <Button
            color="inherit"
            component={RouterLink}
            to="/"
          >
            Sign to Text
          </Button>
          <Button
            color="inherit"
            component={RouterLink}
            to="/text-to-sign"
          >
            Text to Sign
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;
