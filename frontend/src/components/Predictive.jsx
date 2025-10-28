import React, { useEffect, useState, useMemo } from "react";
import {
  Container,
  Card,
  Row,
  Col,
  Button,
  Badge,
  Alert,
  Spinner,
} from "react-bootstrap";
import { motion } from "framer-motion";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Filler,
} from "chart.js";
import { useLiveStream } from "./useLiveStream"; // Updated hook

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Filler
);

const trueRange = (p) => {
  const tr = [];
  for (let i = 1; i < p.length; i++) tr.push(Math.max(p[i] - p[i - 1], 0));
  return tr;
};
const mean = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length || 0;
const gaussianMLE = (r) => {
  // FIXED: Handle empty/insufficient data
  if (r.length < 2) return [0, 0.01];
  const mu = mean(r);
  const variance = r.reduce((s, x) => s + (x - mu) ** 2, 0) / r.length;
  return [mu, Math.sqrt(variance) || 0.01];
};
const erf = (x) => {
  const t = 1.0 / (1.0 + 0.5 * Math.abs(x));
  const tau = t * Math.exp(-x * x - 1.26551223 + 1.00002368 * t + 0.3740916 * t * t + 0.09678418 * t * t * t - 0.18628806 * t * t * t * t + 0.27886807 * t * t * t * t * t - 1.13520398 * t * t * t * t * t * t + 1.48851587 * t * t * t * t * t * t * t - 0.82215223 * t * t * t * t * t * t * t * t + 0.17087277 * t * t * t * t * t * t * t * t * t);
  return x >= 0 ? 1 - tau : tau;
};
const normCDF = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
const nextSuggestion = (mu, sigma) => {
  const sharpe = mu / sigma;
  if (sharpe > 0.2) return { text: "Buy intra-day dips", color: "success" };
  if (sharpe < -0.2) return { text: "Book intraday profit", color: "danger" };
  return { text: "Wait on sidelines", color: "secondary" };
};

export default function Predictive({ symbol: propSymbol = "DABUR.NS" }) {
  // DYNAMIC: Use propSymbol or extract from URL (e.g., /predictive?symbol=SRF.NS)
  const urlParams = new URLSearchParams(window.location.search);
  let urlSymbol = urlParams.get('symbol') || propSymbol;
  // FIXED: Ensure .NS suffix for valid Yahoo Finance symbols
  const symbol = urlSymbol.endsWith('.NS') ? urlSymbol : `${urlSymbol}.NS`;
  const displaySymbol = symbol.replace('.NS', ''); // For title, e.g., "SRF Live Prediction"

  const { ticks, status } = useLiveStream(symbol);
  const [forecast, setForecast] = useState(null);

  useEffect(() => {
    // FIXED: Require min 10 valid ticks
    if (ticks.length < 10) {
      setForecast(null);
      return;
    }
    const prices = ticks.map((x) => x.p).filter(p => Number.isFinite(p) && p > 0);
    if (prices.length < 10) {
      setForecast(null);
      return;
    }
    const returns = prices.slice(1).map((p, i) => Math.log(p / prices[i]));
    // FIXED: Safe MLE
    const [mu, sigma] = gaussianMLE(returns);
    const last = prices[prices.length - 1];
    const nextLog = Math.log(last) + mu;
    const newForecast = {
      mu: Math.exp(nextLog),
      sigma: last * 1.96 * sigma,
      probUp: normCDF(mu / sigma),
      atr: mean(trueRange(prices)),
      suggestion: nextSuggestion(mu, sigma),
    };
    setForecast(newForecast);
  }, [ticks]);

  const chartData = useMemo(() => {
    if (!ticks.length) return { labels: [], datasets: [] };
    const labels = ticks.map((t) => t.t.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}));
    const priceData = ticks.map((t) => t.p);
    const forecastData = [...Array(Math.max(0, ticks.length - 1)).fill(null), forecast?.mu || null];
    return {
      labels,
      datasets: [
        {
          label: "Price",
          data: priceData,
          borderColor: "#667eea",
          backgroundColor: "rgba(102,126,234,0.1)",
          fill: true,
          tension: 0.3,
          pointRadius: 2,
        },
        {
          label: "Forecast",
          data: forecastData,
          borderColor: "#ff6b6b",
          backgroundColor: "transparent",
          borderDash: [5, 5],
          pointRadius: 4,
          borderWidth: 2,
        },
      ],
    };
  }, [ticks, forecast]);

  const chartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: { 
      legend: { display: true, position: 'top' },
      tooltip: {
        mode: 'index',
        intersect: false,
      },
    },
    scales: {
      y: {
        beginAtZero: false,
        ticks: {
          callback: function(value) {
            return '₹' + value.toFixed(2);
          }
        }
      },
      x: {
        title: {
          display: true,
          text: 'Time'
        }
      }
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    }
  }), []);

  if (status === "CONNECTING" && !ticks.length) {
    return (
      <Container fluid className="p-4">
        <Alert variant="info">
          <Spinner animation="border" size="sm" className="me-2" />
          Initializing live stream for {displaySymbol}... Data will load in seconds.
        </Alert>
      </Container>
    );
  }

  if (status === "ERROR" && ticks.length === 0) {
    return (
      <Container fluid className="p-4">
        <Alert variant="danger">
          <i className="fas fa-exclamation-triangle me-2"></i>
          Connection error for {displaySymbol}. Retrying... If persists, check backend server.
        </Alert>
      </Container>
    );
  }

  return (
    <Container fluid className="p-4">
      <motion.div 
        initial={{ opacity: 0, y: 20 }} 
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Row className="justify-content-center g-4">
          <Col md={4}>
            <Card className="shadow-lg border-0 rounded-4 h-100">
              <Card.Body className="d-flex flex-column">
                <div className="d-flex justify-content-between align-items-center mb-3">
                  <h5 className="mb-0 fw-bold text-primary">{displaySymbol} Live Prediction</h5>
                  <Badge bg={status === "LIVE" ? "success" : status === "ERROR" ? "danger" : "warning"} className="fs-6">
                    {status} ({ticks.length} ticks)
                  </Badge>
                </div>
                <h3 className="fw-bold text-primary mb-2">
                  {forecast ? `₹${forecast.mu.toFixed(2)}` : ticks.length > 0 ? "Analyzing..." : "Waiting for data..."}
                </h3>
                <p className="text-muted small mb-1">Next session expected close</p>
                {forecast && (
                  <>
                    <p className="mb-2 small">
                      <span className="fw-semibold">95% Range: </span>
                      ₹{(forecast.mu - forecast.sigma).toFixed(2)} – ₹{(forecast.mu + forecast.sigma).toFixed(2)}
                    </p>
                    <p className="mb-2 small">
                      <span className="fw-semibold">P(↑): </span>{(forecast.probUp * 100).toFixed(1)}%
                    </p>
                    <p className="mb-4 small">
                      <span className="fw-semibold">ATR: </span>₹{forecast.atr.toFixed(2)}
                    </p>
                  </>
                )}
                <Alert
                  variant={forecast?.suggestion?.color || "secondary"}
                  className="flex-grow-1 d-flex align-items-center mb-3"
                >
                  <i className={`fas fa-${forecast?.suggestion?.color === 'success' ? 'arrow-up' : forecast?.suggestion?.color === 'danger' ? 'arrow-down' : 'pause'} me-2`}></i>
                  <strong>Action: </strong> {forecast?.suggestion?.text || "Gathering data..."}
                </Alert>
                <Button
                  variant="outline-primary"
                  size="sm"
                  className="mt-auto w-100"
                  disabled={!ticks.length || !forecast}
                  onClick={() => window.location.href = '/analyze'}
                >
                  <i className="fas fa-chart-line me-2"></i>
                  Deep Analysis
                </Button>
              </Card.Body>
            </Card>
          </Col>
          <Col md={8}>
            <Card className="shadow-lg border-0 rounded-4 h-100">
              <Card.Body className="d-flex flex-column">
                <h5 className="fw-bold mb-3 text-primary">Real-time Price Action & Forecast</h5>
                <div style={{ height: 400, position: 'relative' }}>
                  {ticks.length > 0 ? (
                    <Line
                      data={chartData}
                      options={chartOptions}
                    />
                  ) : (
                    <div className="d-flex justify-content-center align-items-center h-100">
                      <i className="fas fa-chart-line fa-3x text-muted"></i>
                      <p className="ms-3 text-muted">No data yet. Stream starting...</p>
                    </div>
                  )}
                </div>
              </Card.Body>
            </Card>
          </Col>
        </Row>
        <Row className="mt-4">
          <Col>
            <Card className="border-0 rounded-4" bg="light">
              <Card.Body className="text-center text-muted small p-3">
                <i className="fas fa-info-circle me-2"></i>
                This forecast uses a simple Gaussian model on live ticks for {displaySymbol}. Not financial advice—DYOR!
              </Card.Body>
            </Card>
          </Col>
        </Row>
      </motion.div>
    </Container>
  );
}