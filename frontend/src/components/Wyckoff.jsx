// Wyckoff.jsx – Clean, paragraph-based descriptive summary (no numerical counts)
import React from "react";
import { Card, Row, Col, Container } from "react-bootstrap";
import Plot from "react-plotly.js";
import { motion } from "framer-motion";

const Wyckoff = ({ overview, recent, plot }) => {
  if (!overview) {
    return (
      <Container fluid className="p-4 text-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-muted"
        >
          <i className="fas fa-brain fa-3x mb-3"></i>
          <h5>No Wyckoff Data Available</h5>
        </motion.div>
      </Container>
    );
  }

  /* ----------  Helper to form natural summary  ---------- */
  /* ----------  Helper to form natural summary  ---------- */
const summarize = (data, label) => {
  const events = Object.keys(data);
  if (events.length === 0) return "";

  let summaryText = `Wyckoff analysis detects ${label} signals such as `;

  const mapped = events.map((event) => {
    const lower = event.toLowerCase();
    if (lower.includes("strength"))
      return "signs of market strength and accumulation";
    if (lower.includes("spring"))
      return "potential bullish spring formations";
    if (lower.includes("weakness"))
      return "indications of distribution or market weakness";
    if (lower.includes("upthrust"))
      return "possible upthrust traps indicating market exhaustion";
    if (lower.includes("test"))
      return "testing phases validating Wyckoff patterns";
    // fallback: just use the event name descriptively, no numbers
    return `${event} observed in market behavior`;
  });

  summaryText +=
    ". Overall sentiment reflects mixed market conditions with ongoing Wyckoff phase transitions.";

  return summaryText;
};


  const overallSummary = summarize(overview, "overall");
  const recentSummary = summarize(recent, "recent");

  return (
    <Container
      fluid
      className="py-4 px-3"
      style={{
        background: "linear-gradient(180deg, #f8f9ff, #ffffff)",
        minHeight: "100vh",
      }}
    >
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <Row className="g-4">
          {/* ---------- LEFT: Summary Paragraphs ---------- */}
          <Col md={4}>
            <motion.div whileHover={{ scale: 1.01 }}>
              <Card
                className="shadow-lg border-0 rounded-4 h-100"
                style={{
                  background: "linear-gradient(145deg, #ffffff, #f6f7fb)",
                }}
              >
                <Card.Body className="p-4">
                  <Card.Title
                    className="h5 fw-bold text-center mb-4"
                    style={{ color: "#4c51bf" }}
                  >
                    🎯 Wyckoff Pattern Insights
                  </Card.Title>

                  <div
                    className="p-3 bg-light rounded-4 shadow-sm mb-4"
                    style={{ textAlign: "justify", fontSize: "0.93rem" }}
                  >
                    {overallSummary}
                  </div>

                  <h6
                    className="fw-bold text-center mb-3"
                    style={{ color: "#764ba2" }}
                  >
                    Recent Market Activity
                  </h6>

                  <div
                    className="p-3 bg-light rounded-4 shadow-sm"
                    style={{ textAlign: "justify", fontSize: "0.93rem" }}
                  >
                    {recentSummary}
                  </div>
                </Card.Body>
              </Card>
            </motion.div>
          </Col>

          {/* ---------- RIGHT: Wyckoff Chart ---------- */}
          <Col md={8}>
            <motion.div whileHover={{ scale: 1.01 }}>
              <Card
                className="shadow-lg border-0 rounded-4 h-100"
                style={{
                  background: "linear-gradient(145deg, #ffffff, #f6f7fb)",
                }}
              >
                <Card.Body className="p-4">
                  <Card.Title
                    className="h5 fw-bold text-center mb-4"
                    style={{ color: "#4c51bf" }}
                  >
                    📊 Wyckoff Event Distribution
                  </Card.Title>

                  <div
                    className="border bg-light rounded-4 p-2"
                    style={{
                      minHeight: "520px",
                      overflowX: "auto",
                      overflowY: "hidden",
                    }}
                  >
                    {plot && Object.keys(plot).length > 0 ? (
                      <div style={{ width: "720px", height: "500px" }}>
                        <Plot
                          data={plot.data}
                          layout={{
                            ...plot.layout,
                            width: 700,
                            height: 480,
                            margin: { l: 50, r: 40, t: 40, b: 40 },
                            paper_bgcolor: "#ffffff",
                            plot_bgcolor: "#ffffff",
                            font: { size: 12, color: "#222" },
                          }}
                          config={{ responsive: true, displayModeBar: false }}
                          style={{ width: "100%", height: "100%" }}
                        />
                      </div>
                    ) : (
                      <div
                        className="d-flex flex-column justify-content-center align-items-center text-muted"
                        style={{ height: "500px" }}
                      >
                        <i className="fas fa-chart-pie fa-3x mb-3"></i>
                        <small>No Chart Data Available</small>
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

export default Wyckoff;
