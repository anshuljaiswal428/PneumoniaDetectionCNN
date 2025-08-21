import React, { useState } from 'react';
import './Home.css';

function Home() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState('');
  const [error, setError] = useState('');

  const API_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000"; 

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setPrediction('');
    setError('');
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError('Please select a file before submitting.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const result = await response.json();
      console.log(result);
      setPrediction(result.diagnosis);
    } catch (err) {
      console.error('Error fetching prediction:', err);
      setError('Failed to get prediction. Please try again.');
    }
  };

  return (
    <div className="content">
      <h1>Upload X-ray Image</h1>
      <form onSubmit={handleSubmit}>
        <input
          className="input"
          type="file"
          accept="image/*"
          onChange={handleFileChange}
        />
        <br />
        <button className="button" type="submit">
          Submit
        </button>
      </form>

      {prediction && <h2 className="diagnosis">Diagnosis: {prediction}</h2>}
      {error && <p className="error">{error}</p>}
    </div>
  );
}

export default Home;
