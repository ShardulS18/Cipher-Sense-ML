// src/App.js
import { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    // Check if API is running
    fetch('http://localhost:5000/')
      .then(response => {
        if (response.ok) {
          setApiStatus('connected');
        } else {
          setApiStatus('error');
        }
      })
      .catch(() => {
        setApiStatus('error');
      });
  }, []);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a file');
      return;
    }

    setProcessing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      
      if (response.ok) {
        setResult(data);
      } else {
        setError(data.error || 'An error occurred during file processing');
      }
    } catch (err) {
      setError('Failed to connect to the API. Is the server running?');
    } finally {
      setProcessing(false);
    }
  };

  const downloadFile = () => {
    if (result && result.download) {
      window.open(`http://localhost:5000${result.download}`, '_blank');
    }
  };

  return (
    <div className="app-container">
      <div className="card">
        <div className="card-content">
          <div className="header">
            <h1>ML Encryption Tool</h1>
            <div className={`status-indicator ${apiStatus}`}></div>
          </div>
          
          {apiStatus === 'error' && (
            <div className="error-box">
              <p>Cannot connect to the API server. Please make sure it's running at http://localhost:5000</p>
            </div>
          )}

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="file">Select a file to encrypt</label>
              <input
                type="file"
                id="file"
                onChange={handleFileChange}
                className="file-input"
              />
              <p className="help-text">Our ML model will detect the best encryption method based on the file type</p>
            </div>
            
            <div>
              <button
                type="submit"
                disabled={!file || processing || apiStatus !== 'connected'}
                className={`submit-button ${(!file || processing || apiStatus !== 'connected') ? 'disabled' : ''}`}
              >
                {processing ? 'Processing...' : 'Encrypt File'}
              </button>
            </div>
          </form>

          {error && (
            <div className="error-box">
              <p>{error}</p>
            </div>
          )}

          {result && (
            <div className="result-box">
              <h3>Encryption Complete!</h3>
              <div className="result-details">
                <p><span className="label">File:</span> {result.original_filename}</p>
                <p><span className="label">Encryption Method:</span> {result.predicted_cipher}</p>
              </div>
              <button
                onClick={downloadFile}
                className="download-button"
              >
                Download Encrypted File
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;