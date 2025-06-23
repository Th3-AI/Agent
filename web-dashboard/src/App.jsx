import React, { useState, useEffect } from 'react';
import {
  Box, Button, TextField, Typography, Paper, AppBar, Toolbar, IconButton, CircularProgress, List, ListItem, ListItemText, CssBaseline, Container, InputAdornment
} from '@mui/material';
import { Send, PowerSettingsNew, Memory, Terminal, Visibility } from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const API_BASE = 'http://0.0.0.0:8001/api';

const GlassBox = styled(Paper)(({ theme }) => ({
  background: 'rgba(30, 40, 60, 0.7)',
  borderRadius: 20,
  boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
  backdropFilter: 'blur(8px)',
  border: '1px solid rgba(255,255,255,0.18)',
  color: '#fff',
  padding: theme.spacing(3),
}));

const NeonText = styled(Typography)({
  color: '#00fff7',
  textShadow: '0 0 8px #00fff7, 0 0 16px #00fff7',
  fontWeight: 700,
});

function Login({ onLogin }) {
  const [username, setUsername] = useState('admin');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      const formData = new URLSearchParams();
      formData.append('username', username);
      formData.append('password', password);
      const res = await axios.post(`${API_BASE}/login`, formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        }
      });
      onLogin(res.data.access_token);
    } catch (err) {
      setError('Invalid credentials');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box minHeight="100vh" display="flex" alignItems="center" justifyContent="center" sx={{ background: 'radial-gradient(ellipse at top, #23243a 0%, #0f1020 100%)' }}>
      <GlassBox sx={{ minWidth: 350 }}>
        <NeonText variant="h4" align="center" gutterBottom>Agent Dashboard</NeonText>
        <Typography align="center" sx={{ color: '#b0eaff', mb: 2 }}>Sign in to control your agent remotely</Typography>
        <form onSubmit={handleLogin}>
          <TextField
            label="Username"
            value={username}
            onChange={e => setUsername(e.target.value)}
            fullWidth
            margin="normal"
            autoFocus
            InputProps={{ style: { color: '#fff' } }}
          />
          <TextField
            label="Password"
            type="password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            fullWidth
            margin="normal"
            InputProps={{ style: { color: '#fff' } }}
          />
          {error && <Typography color="error" align="center">{error}</Typography>}
          <Button
            type="submit"
            variant="contained"
            color="primary"
            fullWidth
            sx={{ mt: 2, background: 'linear-gradient(90deg,#00fff7,#0051ff)', color: '#111', fontWeight: 700 }}
            disabled={loading}
            endIcon={loading ? <CircularProgress size={20} /> : <PowerSettingsNew />}
          >
            {loading ? 'Signing in...' : 'Sign In'}
          </Button>
        </form>
      </GlassBox>
    </Box>
  );
}

function Dashboard({ token, onLogout }) {
  const [command, setCommand] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [statsHistory, setStatsHistory] = useState([]);
  const [memory, setMemory] = useState({ facts: {}, preferences: {}, conversations: [] });
  const [convRefresh, setConvRefresh] = useState(0);

  // Fetch stats every 3s
  useEffect(() => {
    let interval = setInterval(() => {
      axios.get(`${API_BASE}/stats`, { headers: { Authorization: `Bearer ${token}` } })
        .then(res => {
          setStats(res.data);
          setStatsHistory(h => [...h.slice(-19), { ...res.data, time: new Date().toLocaleTimeString() }]);
        });
    }, 3000);
    return () => clearInterval(interval);
  }, [token]);

  // Fetch memory
  useEffect(() => {
    axios.get(`${API_BASE}/memory`, { headers: { Authorization: `Bearer ${token}` } })
      .then(res => setMemory(res.data));
  }, [token, convRefresh]);

  const handleCommand = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResponse('');
    try {
      const res = await axios.post(`${API_BASE}/command`, { command }, { headers: { Authorization: `Bearer ${token}` } });
      setResponse(res.data.response);
      setConvRefresh(r => r + 1);
    } catch (err) {
      setResponse('Error running command');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ minHeight: '100vh', background: 'radial-gradient(ellipse at top, #23243a 0%, #0f1020 100%)' }}>
      <CssBaseline />
      <AppBar position="static" sx={{ background: 'rgba(20,30,50,0.8)', boxShadow: '0 0 16px #00fff7' }}>
        <Toolbar>
          <NeonText variant="h5" sx={{ flexGrow: 1 }}><Terminal sx={{ mr: 1 }} />Agent Dashboard</NeonText>
          <Button color="inherit" onClick={onLogout} startIcon={<PowerSettingsNew />} sx={{ color: '#00fff7' }}>Logout</Button>
        </Toolbar>
      </AppBar>
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Box display="flex" flexWrap="wrap" gap={4}>
          {/* Command Box */}
          <GlassBox sx={{ flex: 1, minWidth: 340 }}>
            <Typography variant="h6" sx={{ color: '#b0eaff', mb: 1 }}><Terminal sx={{ mr: 1 }} />Run Command</Typography>
            <form onSubmit={handleCommand}>
              <TextField
                label="Type a command..."
                value={command}
                onChange={e => setCommand(e.target.value)}
                fullWidth
                InputProps={{
                  style: { color: '#fff' },
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton type="submit" color="primary" disabled={loading}>
                        <Send />
                      </IconButton>
                    </InputAdornment>
                  )
                }}
                sx={{ mb: 2 }}
              />
            </form>
            {loading && <CircularProgress size={24} sx={{ color: '#00fff7' }} />}
            {response && <Typography sx={{ mt: 2, color: '#fff' }}>{response}</Typography>}
          </GlassBox>

          {/* System Stats */}
          <GlassBox sx={{ flex: 1, minWidth: 340 }}>
            <Typography variant="h6" sx={{ color: '#b0eaff', mb: 1 }}><Visibility sx={{ mr: 1 }} />System Stats</Typography>
            {stats ? (
              <>
                <Typography>OS: <b>{stats.os}</b> ({stats.os_version})</Typography>
                <Box display="flex" gap={2} mt={2}>
                  <StatGauge label="CPU" value={stats.cpu} color="#00fff7" />
                  <StatGauge label="RAM" value={stats.memory} color="#00ffb7" />
                  <StatGauge label="Disk" value={stats.disk} color="#ff00e1" />
                </Box>
                <Box mt={3}>
                  <ResponsiveContainer width="100%" height={120}>
                    <LineChart data={statsHistory}>
                      <XAxis dataKey="time" hide />
                      <YAxis domain={[0, 100]} hide />
                      <CartesianGrid strokeDasharray="3 3" stroke="#222a" />
                      <Tooltip />
                      <Line type="monotone" dataKey="cpu" stroke="#00fff7" dot={false} />
                      <Line type="monotone" dataKey="memory" stroke="#00ffb7" dot={false} />
                      <Line type="monotone" dataKey="disk" stroke="#ff00e1" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              </>
            ) : <CircularProgress sx={{ color: '#00fff7' }} />}
          </GlassBox>

          {/* Memory/Conversation */}
          <GlassBox sx={{ flex: 1, minWidth: 340, maxHeight: 420, overflow: 'auto' }}>
            <Typography variant="h6" sx={{ color: '#b0eaff', mb: 1 }}><Memory sx={{ mr: 1 }} />Agent Memory</Typography>
            <Typography variant="subtitle2" sx={{ color: '#00fff7', mt: 1 }}>Facts:</Typography>
            <List dense>
              {Object.entries(memory.facts || {}).map(([cat, facts]) => facts.map((f, i) => (
                <ListItem key={cat + i}><ListItemText primary={f} /></ListItem>
              )))}
            </List>
            <Typography variant="subtitle2" sx={{ color: '#00fff7', mt: 1 }}>Preferences:</Typography>
            <List dense>
              {Object.entries(memory.preferences || {}).map(([cat, pref]) => (
                <ListItem key={cat}><ListItemText primary={`${cat}: ${pref}`} /></ListItem>
              ))}
            </List>
            <Typography variant="subtitle2" sx={{ color: '#00fff7', mt: 1 }}>Recent Conversations:</Typography>
            <List dense>
              {(memory.conversations || []).map((conv, i) => (
                <ListItem key={i}>
                  <ListItemText
                    primary={<span style={{ color: '#fff' }}>You: <b>{conv.user_input}</b></span>}
                    secondary={<span style={{ color: '#b0eaff' }}>Agent: {conv.agent_response}</span>}
                  />
                </ListItem>
              ))}
            </List>
          </GlassBox>
        </Box>
      </Container>
    </Box>
  );
}

function StatGauge({ label, value, color }) {
  return (
    <Box textAlign="center">
      <svg width="60" height="60">
        <circle cx="30" cy="30" r="26" stroke="#222" strokeWidth="4" fill="none" />
        <circle
          cx="30" cy="30" r="26"
          stroke={color}
          strokeWidth="4"
          fill="none"
          strokeDasharray={2 * Math.PI * 26}
          strokeDashoffset={2 * Math.PI * 26 * (1 - value / 100)}
          style={{ transition: 'stroke-dashoffset 0.5s' }}
        />
        <text x="30" y="36" textAnchor="middle" fontSize="18" fill={color} fontWeight="bold">{Math.round(value)}%</text>
      </svg>
      <Typography variant="caption" sx={{ color }}>{label}</Typography>
    </Box>
  );
}

function App() {
  const [token, setToken] = useState(() => localStorage.getItem('agent_token') || '');

  const handleLogin = (tok) => {
    setToken(tok);
    localStorage.setItem('agent_token', tok);
  };
  const handleLogout = () => {
    setToken('');
    localStorage.removeItem('agent_token');
  };

  return token ? <Dashboard token={token} onLogout={handleLogout} /> : <Login onLogin={handleLogin} />;
}

export default App;
