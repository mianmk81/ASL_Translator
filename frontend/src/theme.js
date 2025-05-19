import { createTheme, alpha } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00f2ff',
      light: '#5cffff',
      dark: '#00bfcc',
      contrastText: '#000',
    },
    secondary: {
      main: '#ff0099',
      light: '#ff56c9',
      dark: '#c6006c',
      contrastText: '#fff',
    },
    background: {
      default: '#0a192f',
      paper: '#112240',
    },
    text: {
      primary: '#e6f1ff',
      secondary: '#8892b0',
    },
    success: {
      main: '#00ffa3',
      light: '#6effc1',
      dark: '#00cc82',
    },
    error: {
      main: '#ff3366',
      light: '#ff6b91',
      dark: '#cc0033',
    },
    info: {
      main: '#00d4ff',
      light: '#6ee7ff',
      dark: '#00a3cc',
    },
  },
  typography: {
    fontFamily: '"Orbitron", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '3.5rem',
      fontWeight: 700,
      letterSpacing: '0.2rem',
      background: 'linear-gradient(45deg, #00f2ff 30%, #ff0099 90%)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
    },
    h2: {
      fontSize: '2.8rem',
      fontWeight: 600,
      letterSpacing: '0.15rem',
    },
    h3: {
      fontSize: '2.2rem',
      fontWeight: 600,
      letterSpacing: '0.1rem',
    },
    h4: {
      fontSize: '1.8rem',
      fontWeight: 500,
      letterSpacing: '0.08rem',
    },
    h5: {
      fontSize: '1.4rem',
      fontWeight: 500,
      letterSpacing: '0.05rem',
    },
    h6: {
      fontSize: '1.2rem',
      fontWeight: 500,
      letterSpacing: '0.04rem',
    },
    body1: {
      fontSize: '1rem',
      letterSpacing: '0.03rem',
      lineHeight: 1.7,
    },
    body2: {
      fontSize: '0.875rem',
      letterSpacing: '0.02rem',
      lineHeight: 1.6,
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          background: 'linear-gradient(135deg, #0a192f 0%, #112240 100%)',
          minHeight: '100vh',
          scrollbarWidth: 'thin',
          '&::-webkit-scrollbar': {
            width: '8px',
          },
          '&::-webkit-scrollbar-track': {
            background: '#112240',
          },
          '&::-webkit-scrollbar-thumb': {
            background: '#00f2ff',
            borderRadius: '4px',
            '&:hover': {
              background: '#5cffff',
            },
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          padding: '10px 24px',
          position: 'relative',
          overflow: 'hidden',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: '-100%',
            width: '100%',
            height: '100%',
            background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent)',
            transition: 'all 0.5s',
          },
          '&:hover::before': {
            left: '100%',
          },
        },
        contained: {
          background: 'linear-gradient(45deg, #00f2ff 30%, #ff0099 90%)',
          boxShadow: '0 3px 15px rgba(0, 242, 255, 0.3)',
          '&:hover': {
            boxShadow: '0 5px 20px rgba(0, 242, 255, 0.5)',
            transform: 'translateY(-2px)',
          },
        },
        outlined: {
          borderColor: '#00f2ff',
          '&:hover': {
            borderColor: '#5cffff',
            boxShadow: '0 0 15px rgba(0, 242, 255, 0.3)',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          background: 'linear-gradient(135deg, #112240 0%, #1a365d 100%)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(0, 242, 255, 0.1)',
          boxShadow: '0 8px 32px rgba(0, 242, 255, 0.1)',
          transition: 'all 0.3s ease-in-out',
          '&:hover': {
            transform: 'translateY(-5px)',
            boxShadow: '0 12px 40px rgba(0, 242, 255, 0.2)',
            border: '1px solid rgba(0, 242, 255, 0.2)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          background: 'linear-gradient(135deg, #112240 0%, #1a365d 100%)',
          backdropFilter: 'blur(10px)',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 8,
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
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          background: 'rgba(17, 34, 64, 0.8)',
          backdropFilter: 'blur(8px)',
          boxShadow: '0 4px 30px rgba(0, 242, 255, 0.1)',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          background: 'linear-gradient(45deg, #00f2ff 30%, #ff0099 90%)',
          color: '#000',
          fontWeight: 500,
        },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: {
          borderRadius: 4,
          background: 'rgba(0, 242, 255, 0.1)',
        },
        bar: {
          borderRadius: 4,
          background: 'linear-gradient(45deg, #00f2ff 30%, #ff0099 90%)',
        },
      },
    },
  },
  transitions: {
    easing: {
      easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
      sharp: 'cubic-bezier(0.4, 0, 0.6, 1)',
      easeOut: 'cubic-bezier(0.0, 0, 0.2, 1)',
      easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
    },
    duration: {
      shortest: 150,
      shorter: 200,
      short: 250,
      standard: 300,
      complex: 375,
      enteringScreen: 225,
      leavingScreen: 195,
    },
  },
});

export default theme;
