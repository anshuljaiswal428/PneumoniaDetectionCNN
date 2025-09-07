import React, { useState } from 'react';
import { Client } from "@gradio/client";
import './Home.css';

function Home() {
  const [file, setFile] = useState(null);
  const [predictionPos, setPredictionPos] = useState('');
  const [predictionNor, setPredictionNor] = useState('');
  const [error, setError] = useState('');

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setPredictionPos('');
    setError('');
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError('Please select a file before submitting.');
      return;
    }

    try {
      const client = await Client.connect("anshul428/pneumonia-detector");

      const result = await client.predict("/predict", {
        pil_img: file,
      });

      const output = result.data[0];
      setPredictionPos(`${output.confidences[0].label} (${(output.confidences[0].confidence * 100).toFixed(1)}%)`);
      setPredictionNor(`${output.confidences[1].label} (${(output.confidences[1].confidence * 100).toFixed(1)}%)`);
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

      {predictionPos && <h2 className="diagnosisInfec">Pneumonia Infection Percentage: {predictionPos}</h2>}
      {predictionNor && <h2 className="diagnosisNorm">No Infection Percentage: {predictionNor}</h2>}
      {error && <p className="error">{error}</p>}
    </div>
  );
}

export default Home;
