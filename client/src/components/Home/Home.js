import React, { useState } from "react";
import { Client } from "@gradio/client";
import "./Home.css";

function Home() {
  const [file, setFile] = useState(null);
  const [predictionPos, setPredictionPos] = useState("");
  const [predictionNor, setPredictionNor] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false); // loader state

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setPredictionPos("");
    setPredictionNor("");
    setError("");
  };

  // helper function to extract confidence number from prediction string
  const getConfidenceValue = (prediction) => {
    if (!prediction) return 0;
    const match = prediction.match(/\(([\d.]+)%\)/); // regex to get percentage
    return match ? parseFloat(match[1]) : 0;
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError("Please select a file before submitting.");
      return;
    }

    setLoading(true); // start loader
    setError("");
    setPredictionPos("");
    setPredictionNor("");

    try {
      const client = await Client.connect("anshul428/pneumonia-detector");

      const result = await client.predict("/predict", {
        pil_img: file,
      });

      const output = result.data[0];

      output.confidences.forEach((conf) => {
        const formatted = `${conf.label} (${(conf.confidence * 100).toFixed(1)}%)`;

        if (conf.label.toLowerCase() === "pneumonia") {
          setPredictionPos(formatted);
        } else if (conf.label.toLowerCase() === "normal") {
          setPredictionNor(formatted);
        }
      });
    } catch (err) {
      console.error("Error fetching prediction:", err);
      setError("Failed to get prediction. Please try again.");
    } finally {
      setLoading(false); // stop loader
    }
  };

  // get numeric values for comparison
  const norValue = getConfidenceValue(predictionNor);
  const posValue = getConfidenceValue(predictionPos);

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
        <button className="button" type="submit" disabled={loading}>
          {loading ? "Loading..." : "Submit"}
        </button>
      </form>

      {/* Loader */}
      {loading && <p className="loader">Processing X-ray, please wait...</p>}

      {predictionPos && (
        <h2 className="diagnosisInfec">
          Pneumonia Infection Percentage: {predictionPos}
        </h2>
      )}
      {predictionNor && (
        <h2 className="diagnosisNorm">
          No Infection Percentage: {predictionNor}
        </h2>
      )}

      {/* Final Diagnosis */}
      {predictionPos && predictionNor && (
        norValue > posValue ? (
          <h2 className="diagnosisNormRes">
            ✅ Person is not having pneumonia ✅
          </h2>
        ) : (
          <h2 className="diagnosisInfecRes">
            ❌ Person is having pneumonia ❌
          </h2>
        )
      )}

      {error && <p className="error">{error}</p>}
    </div>
  );
}

export default Home;
