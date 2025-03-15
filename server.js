
const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const cors = require('cors');

const app = express();

// Enable CORS for all routes
app.use(cors());

// Proxy all requests from /streamlit to the Streamlit server
app.use('/streamlit', createProxyMiddleware({
  target: 'http://localhost:8501',
  changeOrigin: true,
  pathRewrite: {
    '^/streamlit': ''
  },
  onProxyRes: function(proxyRes, req, res) {
    // Remove headers that prevent embedding in iframes
    proxyRes.headers['content-security-policy'] = '';
    proxyRes.headers['x-frame-options'] = '';
  }
}));

// Serve static files
app.use(express.static('dist'));

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Proxy server is running on port ${PORT}`);
});
