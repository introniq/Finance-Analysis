// Trends.jsx
import React from "react";
import { Card, Row, Col, ListGroup, Container, Badge } from "react-bootstrap";
import Plot from "react-plotly.js";
import { motion } from "framer-motion";

const Trends = ({ stats, plot, live }) => {
  /* ---------- Number Formatting Helper ---------- */
  const fmt = (value) => {
    if (isNaN(value)) return "N/A";
    return Number(value).toFixed(2);
  };

  /* ---------- Compute Min / Max from Plot ---------- */
  const pcrArray = (plot?.data?.[0]?.y || []).filter(
    (v) => typeof v === "number"
  );
  const minVal = pcrArray.length ? Math.min(...pcrArray) : 0;
  const maxVal = pcrArray.length ? Math.max(...pcrArray) : 0;

  if (!stats) {
    return (
      <Container fluid className="p-4 text-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6 }}
          className="text-muted"
        >
          <i className="fas fa-trending-up fa-3x mb-3"></i>
          <h5 className="fw-semibold">No Trends Data Available</h5>
        </motion.div>
      </Container>
    );
  }

  const trendColor =
    stats.trend === "Bullish"
      ? "bg-success text-white"
      : "bg-danger text-white";

  return (
    <Container fluid className="py-4 px-3">
      <motion.div
        initial={{ opacity: 0, y: 25 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <Row className="g-4">
          {/* === Left Column: Key Metrics === */}
          <Col md={4}>
            <motion.div whileHover={{ scale: 1.01 }}>
              <Card
                className="shadow-lg border-0 rounded-4 h-100"
                style={{
                  background: "linear-gradient(145deg, #eef2ff, #ffffff)",
                  borderLeft: "6px solid #667eea",
                }}
              >
                <Card.Body className="p-4">
                  <Card.Title
                    className="h5 fw-bold text-center mb-4"
                    style={{ color: "#4c51bf" }}
                  >
                    📊 PCR Key Metrics
                  </Card.Title>

                  <div className="text-center mb-3">
                    <span
                      className="badge bg-info text-dark p-2 px-3"
                      style={{ fontSize: "0.9rem" }}
                    >
                      Live Update: {live?.pcr_update || "N/A"}
                    </span>
                  </div>

                  <ListGroup variant="flush">
                    <ListGroup.Item className="d-flex justify-content-between align-items-center border-0 bg-transparent py-3">
                      <strong>Latest:</strong>
                      <span
                        className={`fw-bold ${
                          stats.latest > 0.3 ? "text-success" : "text-danger"
                        }`}
                      >
                        {fmt(stats.latest)}
                      </span>
                    </ListGroup.Item>

                    <ListGroup.Item className="d-flex justify-content-between align-items-center border-0 bg-transparent py-3">
                      <strong>Mean:</strong>
                      <span className="text-dark">{fmt(stats.mean)}</span>
                    </ListGroup.Item>

                    <ListGroup.Item className="d-flex justify-content-between align-items-center border-0 bg-transparent py-3">
                      <strong>Trend:</strong>
                      <span className={`badge rounded-pill px-3 ${trendColor}`}>
                        {stats.trend || "N/A"}
                      </span>
                    </ListGroup.Item>

                    {/* Min / Max Section */}
                    <ListGroup.Item className="border-0 bg-transparent py-3">
                      <div className="d-flex justify-content-between align-items-center">
                        <strong>Min / Max:</strong>
                        <span className="fw-bold text-dark">
                          {fmt(minVal)} – {fmt(maxVal)}
                        </span>
                      </div>
                      <div className="text-center mt-2">
                        <Badge bg="light" text="dark" className="px-3 py-2">
                          Range: {fmt(maxVal - minVal)}
                        </Badge>
                        <div className="small text-muted mt-1">
                          Derived from chart data
                        </div>
                      </div>
                    </ListGroup.Item>

                    {/* Range Status */}
                    {stats.range_status && (
                      <ListGroup.Item className="text-center border-0 bg-transparent py-3">
                        <small
                          className={`badge rounded-pill px-3 py-2 ${
                            stats.range_status.includes("Optimal")
                              ? "bg-success"
                              : "bg-warning text-dark"
                          }`}
                          style={{ fontSize: "0.85rem" }}
                        >
                          {stats.range_status}
                        </small>
                      </ListGroup.Item>
                    )}
                  </ListGroup>
                </Card.Body>
              </Card>
            </motion.div>
          </Col>

          {/* === Right Column: Chart Visualization === */}
          <Col md={8}>
            <motion.div whileHover={{ scale: 1.01 }}>
              <Card
                className="shadow-lg border-0 rounded-4 h-100"
                style={{
                  background: "linear-gradient(160deg, #ffffff, #f7f8ff)",
                }}
              >
                <Card.Body className="p-4 d-flex flex-column justify-content-between">
                  <Card.Title
                    className="h5 fw-bold text-center mb-4"
                    style={{ color: "#4c51bf" }}
                  >
                    📈 PCR Trend Visualization
                  </Card.Title>

                  <div
                    style={{
                      flexGrow: 1,
                      display: "flex",
                      justifyContent: "center",
                      alignItems: "center",
                      minHeight: "480px",
                    }}
                  >
                    {plot && Object.keys(plot).length > 0 ? (
                      <Plot
                        data={plot.data}
                        layout={{
                          ...plot.layout,
                          height: 480,
                          margin: { l: 50, r: 50, t: 50, b: 50 },
                          paper_bgcolor: "#ffffff",
                          plot_bgcolor: "#ffffff",
                          font: { color: "#1a202c", size: 13 },
                          legend: { orientation: "h", y: -0.2 },
                        }}
                        config={{
                          responsive: true,
                          displayModeBar: true,
                          displaylogo: false,
                        }}
                        style={{
                          width: "100%",
                          height: "100%",
                        }}
                      />
                    ) : (
                      <div className="text-center">
                        <i className="fas fa-chart-area fa-3x text-muted mb-3"></i>
                        <p className="text-secondary fw-medium">
                          No Chart Data Available
                        </p>
                      </div>
                    )}
                  </div>
                </Card.Body>
              </Card>
            </motion.div>
          </Col>
        </Row>
      </motion.div>
    </Container>
  );
};

export default Trends;
