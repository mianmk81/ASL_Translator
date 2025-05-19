import { styled, keyframes } from '@mui/material/styles';
import { Box, Paper, Card, Button } from '@mui/material';

const pulse = keyframes`
  0% {
    box-shadow: 0 0 0 0 rgba(0, 242, 255, 0.4);
  }
  70% {
    box-shadow: 0 0 0 10px rgba(0, 242, 255, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(0, 242, 255, 0);
  }
`;

const float = keyframes`
  0% {
    transform: translateY(0px);
  }
  50% {
    transform: translateY(-10px);
  }
  100% {
    transform: translateY(0px);
  }
`;

export const FuturisticCard = styled(Card)(({ theme }) => ({
  background: 'linear-gradient(135deg, rgba(17, 34, 64, 0.9) 0%, rgba(26, 54, 93, 0.9) 100%)',
  backdropFilter: 'blur(10px)',
  border: '1px solid rgba(0, 242, 255, 0.1)',
  boxShadow: '0 8px 32px rgba(0, 242, 255, 0.1)',
  borderRadius: theme.shape.borderRadius * 2,
  padding: theme.spacing(3),
  transition: 'all 0.3s ease-in-out',
  position: 'relative',
  overflow: 'hidden',
  '&:hover': {
    transform: 'translateY(-5px)',
    boxShadow: '0 12px 40px rgba(0, 242, 255, 0.2)',
    border: '1px solid rgba(0, 242, 255, 0.2)',
  },
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: '2px',
    background: 'linear-gradient(90deg, #00f2ff, #ff0099, #00f2ff)',
    backgroundSize: '200% 100%',
    animation: 'gradient 3s linear infinite',
  },
  '@keyframes gradient': {
    '0%': {
      backgroundPosition: '0% 0%',
    },
    '100%': {
      backgroundPosition: '200% 0%',
    },
  },
}));

export const HologramBox = styled(Box)(({ theme }) => ({
  background: 'linear-gradient(135deg, rgba(0, 242, 255, 0.1) 0%, rgba(255, 0, 153, 0.1) 100%)',
  backdropFilter: 'blur(10px)',
  border: '1px solid rgba(0, 242, 255, 0.2)',
  borderRadius: theme.shape.borderRadius * 2,
  padding: theme.spacing(2),
  position: 'relative',
  overflow: 'hidden',
  '&::after': {
    content: '""',
    position: 'absolute',
    top: '-50%',
    left: '-50%',
    width: '200%',
    height: '200%',
    background: 'linear-gradient(45deg, transparent, rgba(0, 242, 255, 0.1), transparent)',
    transform: 'rotate(30deg)',
    animation: 'hologram 3s linear infinite',
  },
  '@keyframes hologram': {
    '0%': {
      transform: 'rotate(30deg) translateX(-100%)',
    },
    '100%': {
      transform: 'rotate(30deg) translateX(100%)',
    },
  },
}));

export const NeonButton = styled(Button)(({ theme }) => ({
  background: 'transparent',
  border: '2px solid #00f2ff',
  color: '#00f2ff',
  padding: '10px 24px',
  position: 'relative',
  overflow: 'hidden',
  transition: 'all 0.3s ease',
  '&:hover': {
    background: 'rgba(0, 242, 255, 0.1)',
    boxShadow: '0 0 20px rgba(0, 242, 255, 0.5)',
    transform: 'translateY(-2px)',
  },
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: '-100%',
    width: '100%',
    height: '100%',
    background: 'linear-gradient(90deg, transparent, rgba(0, 242, 255, 0.3), transparent)',
    transition: 'all 0.5s',
  },
  '&:hover::before': {
    left: '100%',
  },
}));

export const PulsingDot = styled(Box)(({ theme }) => ({
  width: '12px',
  height: '12px',
  borderRadius: '50%',
  background: '#00f2ff',
  animation: `${pulse} 2s infinite`,
}));

export const FloatingElement = styled(Box)(({ theme }) => ({
  animation: `${float} 3s ease-in-out infinite`,
}));

export const GlassContainer = styled(Box)(({ theme }) => ({
  background: 'rgba(17, 34, 64, 0.4)',
  backdropFilter: 'blur(10px)',
  border: '1px solid rgba(0, 242, 255, 0.1)',
  borderRadius: theme.shape.borderRadius * 2,
  padding: theme.spacing(3),
  position: 'relative',
  overflow: 'hidden',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'radial-gradient(circle at 50% 50%, rgba(0, 242, 255, 0.1), transparent)',
    opacity: 0,
    transition: 'opacity 0.3s ease',
  },
  '&:hover::before': {
    opacity: 1,
  },
}));

export const CyberPanel = styled(Paper)(({ theme }) => ({
  background: 'linear-gradient(135deg, #112240 0%, #1a365d 100%)',
  border: '2px solid rgba(0, 242, 255, 0.2)',
  borderRadius: theme.shape.borderRadius * 2,
  padding: theme.spacing(2),
  position: 'relative',
  '&::before, &::after': {
    content: '""',
    position: 'absolute',
    width: '20px',
    height: '20px',
    border: '2px solid #00f2ff',
  },
  '&::before': {
    top: -2,
    left: -2,
    borderRight: 'none',
    borderBottom: 'none',
  },
  '&::after': {
    bottom: -2,
    right: -2,
    borderLeft: 'none',
    borderTop: 'none',
  },
}));

export const HologramVideo = styled(Box)(({ theme }) => ({
  position: 'relative',
  width: '100%',
  paddingTop: '75%',
  background: '#112240',
  borderRadius: theme.shape.borderRadius * 2,
  overflow: 'hidden',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'linear-gradient(45deg, rgba(0, 242, 255, 0.1), rgba(255, 0, 153, 0.1))',
    mixBlendMode: 'overlay',
  },
  '& video': {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    objectFit: 'cover',
  },
}));

export const DataStream = styled(Box)(({ theme }) => ({
  position: 'relative',
  padding: theme.spacing(2),
  background: 'rgba(17, 34, 64, 0.6)',
  borderRadius: theme.shape.borderRadius,
  border: '1px solid rgba(0, 242, 255, 0.2)',
  fontFamily: 'monospace',
  color: '#00f2ff',
  overflow: 'hidden',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    width: '2px',
    height: '100%',
    background: '#00f2ff',
    animation: 'scan 2s linear infinite',
  },
  '@keyframes scan': {
    '0%': {
      transform: 'translateX(0)',
    },
    '100%': {
      transform: 'translateX(100%)',
    },
  },
}));

export const CircuitBackground = styled(Box)(({ theme }) => ({
  position: 'relative',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'radial-gradient(circle at 50% 50%, rgba(0, 242, 255, 0.1) 1px, transparent 1px)',
    backgroundSize: '20px 20px',
    opacity: 0.5,
  },
}));
