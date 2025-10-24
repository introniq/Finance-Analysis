// Fileuploader.jsx
import React, { useState, useRef, useEffect } from "react";
import {
  Container,
  Row,
  Col,
  Card,
  Form,
  Button,
  OverlayTrigger,
  Tooltip,
  Alert,
  Spinner
} from "react-bootstrap";
import { motion } from "framer-motion";
import Analysis from "./Analysis";

export default function FileAnalysisUploader({ historyFile = null }) {
  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [params, setParams] = useState({
    algorithm: "KMeans",
    daysAhead: "10",
    windowSize: "10",
    clusters: "3",
    rsiPeriod: "14",
    adxPeriod: "14",
  });
  const [showResults, setShowResults] = useState(false);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const uploadRef = useRef(null);
  const API_BASE = "http://localhost:8050";

  const handleFile = (files) => {
    if (files && files[0]) {
      setFile(files[0]);
      setError(null);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
    else if (e.type === "dragleave") setDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0])
      handleFile(e.dataTransfer.files);
  };

  const handleChangeFile = (e) => handleFile(e.target.files);
  const handleChangeParams = (e) =>
    setParams({ ...params, [e.target.name]: e.target.value });

  const handleSubmit = async (e) => {
  e.preventDefault();
  if (!file) {
    setError("Please select a file.");
    return;
  }
  setLoading(true);
  setError(null);

  const formData = new FormData();
  formData.append("file", file);
  formData.append("params", JSON.stringify(params));

  try {
    // 1️⃣ Analyze request
    const response = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    const data = await response.json();
    if (data.error) throw new Error(data.error);
    setResults(data);
    setShowResults(true);

    // 2️⃣ Save file to server history
    const saveResponse = await fetch(`http://localhost:8051/save-file`, {
      method: "POST",
      body: formData,
    });
    if (!saveResponse.ok) throw new Error(`Save failed: ${saveResponse.statusText}`);
  } catch (err) {
    setError(`Analysis failed: ${err.message}`);
  } finally {
    setLoading(false);
  }
};

useEffect(() => {
    if (historyFile) {
      runAnalysisFromHistory(historyFile);
    }
  }, [historyFile]);

  const runAnalysisFromHistory = async (filename) => {
    setLoading(true);
    setShowResults(false);
    setResults(null);
    try {
      // fetch analysis for existing file
      const response = await fetch(`${API_BASE}/analyze?file=${filename}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setResults(data);
      setShowResults(true);
      setFile({ name: filename }); // just for display
    } catch (err) {
      console.error("Analysis failed:", err);
    } finally {
      setLoading(false);
    }
  };

  const renderTooltip = (text) => <Tooltip>{text}</Tooltip>;

  if (error && !loading) {
    return (
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="container mt-4">
        <Alert variant="danger" onClose={() => setError(null)} dismissible className="shadow-lg">
          <div className="d-flex align-items-center">
            <i className="fas fa-exclamation-triangle me-2"></i>
            {error}
          </div>
        </Alert>
      </motion.div>
    );
  }

  return (
    <Container
      fluid
      style={{
        height: "88vh",
        marginTop: "1vh",
        backgroundColor: "#f8f9fa",
        borderRadius: "12px",
        boxShadow: "0 4px 20px rgba(0,0,0,0.05)",
        border: "1px solid #dcdcdc",
        overflowY: "scroll",
        overflowX: "hidden",
        scrollbarWidth: "none",
        msOverflowStyle: "none",
      }}
    >
      <Row className="justify-content-center gap-3 align-items-stretch">
        {!showResults ? (
          <>
            {/* File Upload Card */}
            <Col xs={12} md={8}>
              <Card
                ref={uploadRef}
                className={`text-center p-3 rounded-4 border-2 shadow-sm d-flex align-items-center justify-content-center ${
                  dragActive ? "border-primary bg-light" : "border-secondary bg-white"
                }`}
                onDragEnter={handleDrag}
                onDragOver={handleDrag}
                onDragLeave={handleDrag}
                onDrop={handleDrop}
                style={{
                  transition: "all 0.3s",
                  cursor: "pointer",
                  minHeight: "200px",
                  marginTop: "2vh",
                }}
              >
                <input
                  type="file"
                  id="file-upload"
                  onChange={handleChangeFile}
                  style={{ display: "none" }}
                  accept=".csv,.xlsx"
                />
                <label
                  htmlFor="file-upload"
                  className="d-flex flex-column align-items-center w-100 h-100 justify-content-center"
                >
                  <div
                    className="rounded-circle d-flex align-items-center justify-content-center mb-2"
                    style={{
                      width: "90px",
                      height: "90px",
                      backgroundColor: dragActive ? "#cce5ff" : "#e9ecef",
                      transition: "all 0.3s",
                    }}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                      strokeWidth={1.5}
                      stroke={dragActive ? "#0d6efd" : "#6c757d"}
                      width="36"
                      height="36"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M3 16.5V19a2.25 2.25 0 002.25 2.25h13.5A2.25 2.25 0 0021 19v-2.5M7.5 10.5L12 6m0 0l4.5 4.5M12 6v12"
                      />
                    </svg>
                  </div>
                  {file ? (
                    <p className="fw-medium text-truncate" style={{ maxWidth: "160px" }}>
                      📄 {file.name}
                    </p>
                  ) : (
                    <>
                      <p className="fw-bold mb-1">Drag & Drop Your File</p>
                      <p className="text-muted mb-0">
                        or <span className="text-primary text-decoration-underline">Browse</span>
                      </p>
                    </>
                  )}
                </label>
              </Card>
            </Col>

            {/* Analysis Parameters Card */}
            <Col xs={12} md={8}>
              <Card className="p-3 rounded-4 shadow-sm border-1 border-light h-100">
                <Card.Title className="mb-3 fw-bold border-bottom pb-1">Analysis Parameters</Card.Title>
                <Form onSubmit={handleSubmit} className="gap-2 d-flex flex-column">
                  <Row className="mb-2">
                    <Col>
                      <Form.Group>
                        <Form.Label>Clustering Algorithm</Form.Label>
                        <OverlayTrigger overlay={renderTooltip("Select the clustering method for grouping data")}>
                          <Form.Select
                            name="algorithm"
                            value={params.algorithm}
                            onChange={handleChangeParams}
                          >
                            <option>KMeans</option>
                          </Form.Select>
                        </OverlayTrigger>
                      </Form.Group>
                    </Col>
                    <Col>
                      <Form.Group>
                        <Form.Label>Prediction Horizon (Days Ahead)</Form.Label>
                        <OverlayTrigger overlay={renderTooltip("Number of days ahead for price prediction")}>
                          <Form.Control
                            type="number"
                            name="daysAhead"
                            value={params.daysAhead}
                            onChange={handleChangeParams}
                          />
                        </OverlayTrigger>
                      </Form.Group>
                    </Col>
                  </Row>

                  <Row className="mb-2">
                    <Col>
                      <Form.Group>
                        <Form.Label>Window Size (Days)</Form.Label>
                        <OverlayTrigger overlay={renderTooltip("Number of past days to use for rolling analysis")}>
                          <Form.Control
                            type="number"
                            name="windowSize"
                            value={params.windowSize}
                            onChange={handleChangeParams}
                          />
                        </OverlayTrigger>
                      </Form.Group>
                    </Col>
                    <Col>
                      <Form.Group>
                        <Form.Label>Number of Clusters</Form.Label>
                        <OverlayTrigger overlay={renderTooltip("Applicable for KMeans/Hierarchical clustering")}>
                          <Form.Control
                            type="number"
                            name="clusters"
                            value={params.clusters}
                            onChange={handleChangeParams}
                          />
                        </OverlayTrigger>
                      </Form.Group>
                    </Col>
                  </Row>

                  <Row className="mb-2">
                    <Col>
                      <Form.Group>
                        <Form.Label>RSI Period</Form.Label>
                        <OverlayTrigger overlay={renderTooltip("Number of periods for RSI calculation")}>
                          <Form.Control
                            type="number"
                            name="rsiPeriod"
                            value={params.rsiPeriod}
                            onChange={handleChangeParams}
                          />
                        </OverlayTrigger>
                      </Form.Group>
                    </Col>
                    <Col>
                      <Form.Group>
                        <Form.Label>ADX Period</Form.Label>
                        <OverlayTrigger overlay={renderTooltip("Number of periods for ADX calculation")}>
                          <Form.Control
                            type="number"
                            name="adxPeriod"
                            value={params.adxPeriod}
                            onChange={handleChangeParams}
                          />
                        </OverlayTrigger>
                      </Form.Group>
                    </Col>
                  </Row>

                  <Button
                    type="submit"
                    className="mt-2 w-100 fw-bold"
                    style={{
                      backgroundColor: "#0d6efd",
                      border: "none",
                      height: "45px",
                    }}
                    disabled={loading}
                  >
                    {loading ? (
                      <>
                        <Spinner animation="border" size="sm" className="me-2" />
                        Running Analysis...
                      </>
                    ) : (
                      "Run Analysis"
                    )}
                  </Button>
                </Form>
              </Card>
            </Col>
          </>
        ) : (
          <Analysis results={results} file={file} />
        )}
      </Row>
    </Container>
  );
}
