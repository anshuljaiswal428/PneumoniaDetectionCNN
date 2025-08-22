import React, { useState } from 'react';
import { Client } from "@gradio/client";
import './Home.css';

function Home() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState('');
  const [error, setError] = useState('');

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

    try {
      // Connect to your Hugging Face Space
      const client = await Client.connect("anshul428/pneumonia-detector");

      // Call /predict with your uploaded file
      const result = await client.predict("/predict", {
        pil_img: file,   // key name matches your `app.py`
      });

      console.log("Raw result:", result.data);

      // Gradio Label output looks like: { "label": "Pneumonia", "confidence": 0.92 }
      const output = result.data[0];
      setPrediction(`${output.label} (${(output.confidence * 100).toFixed(1)}%)`);
    } catch (err) {
      console.error("Error fetching prediction:", err);
      setError("Failed to get prediction. Please try again.");
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
