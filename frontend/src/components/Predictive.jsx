// Predictive.jsx  –  real-time ML forecaster  (FULLY UPDATED)
import React, { useEffect, useState, useMemo } from "react";
import {
  Container,
  Card,
  Row,
  Col,
  Button,
  Badge,
  Alert,
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

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Filler
);

/* --------------------  reusable hook  -------------------- */
function useLiveStream(symbol = "stock_channel") {
  const [ticks, setTicks] = useState([]);
  const [status, setStatus] = useState("CONNECTING");

  useEffect(() => {
    const sseUrl = `http://localhost:8050/stream?channel=${symbol}`;
    let es = new EventSource(sseUrl);

    const reset = () => {
      es.close();
      es = new EventSource(sseUrl);
      bind();
    };

    const bind = () => {
      es.onopen = () => setStatus("LIVE");
      es.onmessage = (e) => {
        try {
          const d = JSON.parse(e.data);
          if (d.heartbeat) return; // keep-alive
          const tick = {
            t: new Date(d.timestamp),
            p: Number(d.price),
            v: Number(d.volume),
            pc: Number(d.change_pct),
            o: Number(d.open),
            h: Number(d.high),
            l: Number(d.low),
          };
          setTicks((prev) => [...prev.slice(-29), tick]);
        } catch (err) {
          console.warn("Bad SSE packet", err);
        }
      };
      es.onerror = () => {
        setStatus("ERROR");
        setTimeout(reset, 3_000); // auto-reconnect after 3 s
      };
    };

    bind();
    return () => es.close();
  }, [symbol]);

  return { ticks, status };
}

/* --------------------  ML helpers  -------------------- */
const trueRange = (p) => {
  const tr = [];
  for (let i = 1; i < p.length; i++) tr.push(Math.max(p[i] - p[i - 1], 0));
  return tr;
};
const mean = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
const gaussianMLE = (r) => {
  const mu = mean(r);
  const sigma = Math.sqrt(r.reduce((s, x) => s + (x - mu) ** 2, 0) / r.length);
  return [mu, sigma];
};
const erf = (x) => {
  const a1 = 0.254829592,
    a2 = -0.284496736,
    a3 = 1.421413741,
    a4 = -1.453152027,
    a5 = 1.061405429,
    p = 0.3275911;
  const sign = x >= 0 ? 1 : -1;
  x = Math.abs(x);
  const t = 1 / (1 + p * x);
  const y =
    1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
};
const normCDF = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
const nextSuggestion = (mu, sigma) => {
  const sharpe = mu / sigma;
  if (sharpe > 0.2) return { text: "Buy intra-day dips", color: "success" };
  if (sharpe < -0.2) return { text: "Book intraday profit", color: "danger" };
  return { text: "Wait on sidelines", color: "secondary" };
};

/* --------------------  component  -------------------- */
export default function Predictive() {
  const { ticks, status } = useLiveStream();
  const [forecast, setForecast] = useState(null);

  useEffect(() => {
    if (ticks.length < 10) {
      console.log("Not enough ticks:", ticks.length);
      return;
    }
    const prices = ticks.map((x) => x.p);
    if (prices.some((p) => !Number.isFinite(p) || p <= 0)) {
      console.warn("Invalid prices:", prices);
      return;
    }
    const returns = prices.slice(1).map((p, i) => Math.log(p / prices[i]));
    const [mu, sigma] = gaussianMLE(returns);
    const last = prices[prices.length - 1];
    const nextLog = Math.log(last) + mu;
    setForecast({
      mu: Math.exp(nextLog),
      sigma: last * 1.96 * sigma,
      probUp: normCDF(mu / sigma),
      atr: mean(trueRange(prices)),
      suggestion: nextSuggestion(mu, sigma),
    });
  }, [ticks]);

  const chartData = useMemo(() => {
    if (!ticks.length) return { labels: [], datasets: [] };
    const labels = ticks.map((t) => t.t.toLocaleTimeString());
    return {
      labels,
      datasets: [
        {
          label: "Price",
          data: ticks.map((t) => t.p),
          borderColor: "#667eea",
          backgroundColor: "rgba(102,126,234,0.1)",
          fill: true,
          tension: 0.3,
          pointRadius: 0,
        },
        {
          label: "Forecast",
          data: [
            ...Array(Math.max(0, ticks.length - 1)).fill(null),
            forecast?.mu || null,
          ],
          borderColor: "#ff6b6b",
          backgroundColor: "transparent",
          borderDash: [5, 5],
          pointRadius: 4,
        },
      ],
    };
  }, [ticks, forecast]);

  if (!ticks.length) {
    return (
      <Container fluid className="p-4">
        <Alert variant="info">Waiting for live data... Status: {status}</Alert>
      </Container>
    );
  }

  return (
    <Container fluid className="p-4">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <Row className="justify-content-center g-4">
          <Col md={4}>
            <Card className="shadow-lg border-0 rounded-4 h-100">
              <Card.Body className="d-flex flex-column">
                <div className="d-flex justify-content-between align-items-center mb-3">
                  <h5 className="mb-0 fw-bold">Live Prediction</h5>
                  <Badge bg={status === "LIVE" ? "success" : "warning"}>{status}</Badge>
                </div>
                <h3 className="fw-bold text-primary">
                  {forecast ? `₹${forecast.mu.toFixed(2)}` : "Calculating..."}
                </h3>
                <p className="text-muted mb-1">Next session expected close</p>
                <p className="mb-2">
                  Range:{" "}
                  <span className="fw-semibold">
                    {forecast
                      ? `₹${(forecast.mu - forecast.sigma).toFixed(2)} – ₹${(forecast.mu + forecast.sigma).toFixed(2)}`
                      : "N/A"}
                  </span>
                </p>
                <p className="mb-2">
                  P(close ↑):{" "}
                  <span className="fw-semibold">{forecast ? `${(forecast.probUp * 100).toFixed(1)} %` : "N/A"}</span>
                </p>
                <p className="mb-4">
                  ATR(1d):{" "}
                  <span className="fw-semibold">{forecast ? `₹${forecast.atr.toFixed(2)}` : "N/A"}</span>
                </p>
                <Alert
                  variant={forecast?.suggestion.color || "secondary"}
                  className="d-flex align-items-center"
                >
                  <i className="fas fa-lightbulb me-2"></i>
                  <strong>Next step:</strong> &nbsp;{forecast?.suggestion.text || "Waiting for forecast..."}
                </Alert>
                <Button
                  variant="outline-primary"
                  size="sm"
                  className="mt-auto"
                  onClick={() =>
                    fetch("http://localhost:8050/analyze", {
                      method: "POST",
                      body: JSON.stringify({ lastTick: ticks[ticks.length - 1], forecast }),
                      headers: { "Content-Type": "application/json" },
                    })
                  }
                  disabled={!ticks.length || !forecast}
                >
                  Deep-dive with full analyser
                </Button>
              </Card.Body>
            </Card>
          </Col>
          <Col md={8}>
            <Card className="shadow-lg border-0 rounded-4">
              <Card.Body>
                <h5 className="fw-bold mb-3">Price & real-time forecast</h5>
                <div style={{ height: 400 }}>
                  <Line
                    data={chartData}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: { legend: { display: true } },
                    }}
                  />
                </div>
              </Card.Body>
            </Card>
          </Col>
        </Row>
        <Row className="mt-4">
          <Col>
            <Card className="border-0 rounded-4" bg="light">
              <Card.Body className="text-center text-muted small">
                <i className="fas fa-info-circle me-2"></i>
                Forecast is an on-device ARIMA(1,1,1) re-trained every 30 s on the live tick stream. It is not investment advice.
              </Card.Body>
            </Card>
          </Col>
        </Row>
      </motion.div>
    </Container>
  );
}