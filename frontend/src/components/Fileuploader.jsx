// Fileuploader.jsx
import React, { useState, useRef } from "react";
import { Container, Row, Col, Card, Form, Button, OverlayTrigger, Tooltip, Alert, Spinner } from "react-bootstrap";
import Analysis from "./Analysis";
import { motion } from 'framer-motion';

export default function FileAnalysisUploader() {
  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [params, setParams] = useState({
    algorithm: "kmeans",
    daysAhead: "10",
    windowSize: "10",
    clusters: "3",
    rsiPeriod: "14",
    adxPeriod: "14"
  });
  const [showResults, setShowResults] = useState(false);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const uploadRef = useRef(null);
  const API_BASE = 'http://localhost:8050';

  const handleFile = (files) => { 
    if (files && files[0]) {
      setFile(files[0]);
      setError(null); // Clear any previous errors
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
    if (e.dataTransfer.files && e.dataTransfer.files[0]) handleFile(e.dataTransfer.files);
  };

  const handleChangeFile = (e) => handleFile(e.target.files);

  const handleChangeParams = (e) => setParams({ ...params, [e.target.name]: e.target.value });

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a file.");
      return;
    }
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('params', JSON.stringify(params));
    try {
      const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        body: formData
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      const data = await response.json();
      if (data.error) throw new Error(data.error);
      setResults(data);
      setShowResults(true);
    } catch (err) {
      setError(`Analysis failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const renderTooltip = (text) => <Tooltip id="tooltip" placement="top">{text}</Tooltip>;

  if (error && !loading) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="container mt-4"
      >
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
    <Container fluid style={{ height: '88vh', marginTop: '1vh', background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)', borderRadius: "12px", boxShadow: "0 8px 32px rgba(0,0,0,0.1)", overflow: 'hidden' }}>
      <Row className="h-100 justify-content-center align-items-stretch">
        {!showResults ? (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="col-12 d-flex flex-column justify-content-center align-items-center p-4"
          >
            <Col xs={12} md={8} className="mb-4">
              <Card
                ref={uploadRef}
                className={`text-center p-5 rounded-4 border-0 shadow-lg d-flex align-items-center justify-content-center transition-all ${dragActive ? "bg-primary bg-opacity-10 border-primary" : "bg-white"}`}
                onDragEnter={handleDrag}
                onDragOver={handleDrag}
                onDragLeave={handleDrag}
                onDrop={handleDrop}
                style={{ transition: "all 0.3s ease", cursor: "pointer", minHeight: "300px", transform: dragActive ? 'scale(1.02)' : 'scale(1)' }}
              >
                <input type="file" id="file-upload" onChange={handleChangeFile} style={{ display: "none" }} accept=".csv,.xlsx" />
                <label htmlFor="file-upload" className="w-100 h-100 d-flex flex-column justify-content-center align-items-center">
                  {dragActive ? (
                    <div className="text-primary">
                      <i className="fas fa-cloud-upload-alt fa-3x mb-3"></i>
                      <h5 className="mb-2">Drop your file here!</h5>
                      <p className="text-muted">CSV or XLSX files supported</p>
                    </div>
                  ) : (
                    <div>
                      <i className="fas fa-cloud-upload-alt fa-3x mb-3 text-muted"></i>
                      <h5 className="mb-2 text-muted">Drag & Drop or Click to Upload</h5>
                      <p className="text-muted">Upload your stock analysis CSV/XLSX file</p>
                      {file && <p className="mt-2 bg-success bg-opacity-10 p-2 rounded text-success">{file.name}</p>}
                    </div>
                  )}
                </label>
              </Card>
            </Col>
            <Col xs={12} md={8}>
              <Card className="shadow-lg border-0 rounded-4 p-4" style={{ background: 'linear-gradient(145deg, #ffffff, #f8f9fa)' }}>
                <Form onSubmit={handleSubmit}>
                  <Row>
                    <Col md={6}>
                      <Form.Group className="mb-3">
                        <Form.Label className="fw-bold">Clustering Algorithm</Form.Label>
                        <OverlayTrigger overlay={renderTooltip("Choose clustering method for pattern analysis")}>
                          <Form.Select name="algorithm" value={params.algorithm} onChange={handleChangeParams} className="rounded-3">
                            <option value="kmeans">KMeans</option>
                          </Form.Select>
                        </OverlayTrigger>
                      </Form.Group>
                    </Col>
                    <Col md={6}>
                      <Form.Group className="mb-3">
                        <Form.Label className="fw-bold">Prediction Horizon (Days)</Form.Label>
                        <OverlayTrigger overlay={renderTooltip("Number of days ahead for price prediction")}>
                          <Form.Control type="number" name="daysAhead" value={params.daysAhead} onChange={handleChangeParams} className="rounded-3" min="1" />
                        </OverlayTrigger>
                      </Form.Group>
                    </Col>
                  </Row>
                  <Row className="mb-3">
                    <Col md={6}>
                      <Form.Group>
                        <Form.Label className="fw-bold">Window Size (Days)</Form.Label>
                        <OverlayTrigger overlay={renderTooltip("Number of past days to use for rolling analysis")}>
                          <Form.Control type="number" name="windowSize" value={params.windowSize} onChange={handleChangeParams} className="rounded-3" min="2" />
                        </OverlayTrigger>
                      </Form.Group>
                    </Col>
                    <Col md={6}>
                      <Form.Group>
                        <Form.Label className="fw-bold">Number of Clusters</Form.Label>
                        <OverlayTrigger overlay={renderTooltip("Applicable for KMeans/Hierarchical clustering")}>
                          <Form.Control type="number" name="clusters" value={params.clusters} onChange={handleChangeParams} className="rounded-3" min="2" />
                        </OverlayTrigger>
                      </Form.Group>
                    </Col>
                  </Row>
                  <Row className="mb-3">
                    <Col md={6}>
                      <Form.Group>
                        <Form.Label className="fw-bold">RSI Period</Form.Label>
                        <OverlayTrigger overlay={renderTooltip("Number of periods for Relative Strength Index calculation")}>
                          <Form.Control type="number" name="rsiPeriod" value={params.rsiPeriod} onChange={handleChangeParams} className="rounded-3" min="5" />
                        </OverlayTrigger>
                      </Form.Group>
                    </Col>
                    <Col md={6}>
                      <Form.Group>
                        <Form.Label className="fw-bold">ADX Period</Form.Label>
                        <OverlayTrigger overlay={renderTooltip("Number of periods for Average Directional Index calculation")}>
                          <Form.Control type="number" name="adxPeriod" value={params.adxPeriod} onChange={handleChangeParams} className="rounded-3" min="5" />
                        </OverlayTrigger>
                      </Form.Group>
                    </Col>
                  </Row>
                  <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                    <Button type="submit" className="mt-3 w-100 rounded-3 fw-bold" style={{ background: 'linear-gradient(45deg, #667eea, #764ba2)', border: "none", height: '50px' }} disabled={loading}>
                      {loading ? (
                        <>
                          <Spinner animation="border" size="sm" className="me-2" />
                          Running Advanced Analysis...
                        </>
                      ) : (
                        <>
                          <i className="fas fa-rocket me-2"></i>
                          Launch Analysis
                        </>
                      )}
                    </Button>
                  </motion.div>
                </Form>
              </Card>
            </Col>
          </motion.div>
        ) : (
          <Col xs={12} className="p-0 h-100">
            {/* FIXED: Pass file prop to Analysis for downstream components like Volume */}
            <Analysis results={results} file={file} />
          </Col>
        )}
      </Row>
    </Container>
  );
}