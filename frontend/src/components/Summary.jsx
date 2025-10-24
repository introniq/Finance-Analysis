// Summary.jsx
import React from 'react';
import { ListGroup, Container, Row, Col, Card, Badge, Alert } from 'react-bootstrap';
import { motion } from 'framer-motion';

const Summary = ({ data, live }) => {
  /* ----------  early-exit skeleton  ---------- */
  if (!data)
    return (
      <Container fluid className="d-flex align-items-center justify-content-center vh-100">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center text-muted"
        >
          <div className="spinner-border text-primary mb-3" role="status" />
          <h5 className="fw-bold">No summary data</h5>
          <p>Run the analysis to populate the dashboard.</p>
        </motion.div>
      </Container>
    );

  /* ----------  derived values  ---------- */
  const { metrics, delivery_check, outlook, pattern } = data;
  const price = live?.price || metrics.current_price;

  /* ----------  tiny reusable metric tile  ---------- */
  const MetricTile = ({ icon, label, value, color }) => (
    <Card className="shadow-sm h-100 border-0" bg={color} text={color==='dark' ? 'white' : 'dark'}>
      <Card.Body className="d-flex flex-column align-items-center justify-content-center p-3">
        <div className="fs-2 mb-2">{icon}</div>
        <div className="fw-semibold small text-center">{label}</div>
        <div className="fw-bold fs-5 mt-1">{value}</div>
      </Card.Body>
    </Card>
  );

  /* ----------  main render  ---------- */
  return (
    <Container fluid className="px-3 ">
        {/* ----- top KPI row ----- */}
        <Row className="g-3 mb-4 ">
          <Col xs={6} md={2}><MetricTile icon="🏢" label="Symbol" value={metrics.symbol||'—'} color="light" /></Col>
          <Col xs={6} md><MetricTile icon="👥" label="Shares Outstanding" value={metrics.outstanding_shares?.toLocaleString()||'—'} color="light" /></Col>
          <Col xs={6} md><MetricTile icon="💰" label="Current Price" value={price ? `₹${Number(price).toFixed(2)}` : '—'} color="light" /></Col>
          <Col xs={6} md><MetricTile icon="🏛️" label="Market-Cap (Cr)" value={metrics.market_cap_cr||'—'} color="light" /></Col>
          <Col xs={6} md><MetricTile icon="📊" label="52-Wk Range" value={metrics['52w_high_low']||'—'} color="dark" /></Col>
        </Row>

        {/* ----- delivery alert ----- */}
        <Alert variant={delivery_check.color} className="d-flex align-items-center mb-4 py-2">
          <i className={`me-2 fs-5 fa fa-${delivery_check.color==='success'?'check-circle':'exclamation-triangle'}`} />
          <div>
            <span className="fw-bold d-block">{delivery_check.message}</span>
            {delivery_check.delivery_status && <small className="d-block text-muted">{delivery_check.delivery_status}</small>}
            {delivery_check.oi_status && <small className="d-block text-muted">{delivery_check.oi_status}</small>}
            <small className="d-block">– Delivery & OI status (last 10 days)</small>
          </div>
        </Alert>

        {/* ----- two symmetric columns ----- */}
        <Row className="g-4">
          {/* left – market outlook */}
          <Col lg={6} className="d-flex">
            <Card className="shadow w-100 border-0">
              <Card.Header className="bg-primary text-white fw-bold">Market Outlook</Card.Header>
              <Card.Body className="p-3 d-flex flex-column justify-content-between">
                <Row className="g-3 mb-3">
                  <Col xs={6}><Kpi label="Latest Close"        val={outlook.close}        sign="₹" /></Col>
                  <Col xs={6}><Kpi label="5-Day MA"            val={outlook.ma5}          sign="₹" /></Col>
                  <Col xs={6}><Kpi label="PCR (curr)"          val={outlook.pcr}          after={live?.pcr_update && <Badge bg="success" className="ms-2">{live.pcr_update.toFixed(2)}</Badge>} /></Col>
                  <Col xs={6}><Kpi label="PCR 5-D Avg"         val={outlook.pcr_ma5} /></Col>
                </Row>

                <div className="text-center mt-auto">
                  <h5 className={`fw-bold mb-2 ${outlook.wyckoff_outlook==='Bullish'?'text-success':outlook.wyckoff_outlook==='Bearish'?'text-danger':'text-warning'}`}>
                    <i className={`fa fa-arrow-${outlook.wyckoff_outlook==='Bullish'?'up':outlook.wyckoff_outlook==='Bearish'?'down':'minus'} me-2`} />
                    {outlook.wyckoff_event||'Neutral'} – {outlook.wyckoff_outlook||'Neutral'} bias
                  </h5>
                  <Badge bg={oiColor(outlook.oi_divergence)} className="px-3 py-2">{outlook.oi_divergence||'No divergence'}</Badge>
                </div>
              </Card.Body>
            </Card>
          </Col>

          {/* right – pattern recognition */}
          <Col lg={6} className="d-flex">
            <Card className="shadow w-100 border-0">
              <Card.Header className="bg-secondary text-white fw-bold">Pattern & Guidance</Card.Header>
              <Card.Body className="p-3 d-flex flex-column justify-content-between">
                <ListGroup variant="flush" className="border-0">
                  <ListGroup.Item className="d-flex justify-content-between px-0">
                    <span><i className="fa fa-list text-info me-2" />Recent FOI (last 10)</span>
                    <Badge bg="info" className="px-2">{pattern.recent_foi?.length >= 10 ? pattern.recent_foi.map(p => p.toFixed(2)).join(', ') : '—'}</Badge>
                  </ListGroup.Item>
                  <ListGroup.Item className="d-flex justify-content-between px-0">
                    <span><i className="fa fa-layer-group text-primary me-2" />Cluster</span>
                    <Badge bg="primary">{pattern.cluster_match||'—'}</Badge>
                  </ListGroup.Item>
                  <ListGroup.Item className="d-flex justify-content-between px-0">
                    <span><i className="fa fa-percentage text-warning me-2" />Exp. Return</span>
                    <Badge bg={pattern.expected_change>0?'success':'danger'}>{pattern.expected_change>0?'+':''}{pattern.expected_change?.toFixed(1)||'—'}%</Badge>
                  </ListGroup.Item>
                  <ListGroup.Item className="d-flex justify-content-between px-0">
                    <span><i className="fa fa-compass text-success me-2" />Guidance</span>
                    <Badge bg={pattern.guidance==='Bullish'?'success':pattern.guidance==='Bearish'?'danger':'warning'}>{pattern.guidance||'Neutral'}</Badge>
                  </ListGroup.Item>
                  <ListGroup.Item className="d-flex justify-content-between px-0">
                    <span><i className="fa fa-eye text-secondary me-2" />Wyckoff</span>
                    <Badge bg="dark">{pattern.wyckoff_event||'Neutral'}</Badge>
                  </ListGroup.Item>
                </ListGroup>
              </Card.Body>
            </Card>
          </Col>
        </Row>
    </Container>
  );
};

/* ----------  tiny helpers  ---------- */
const Kpi = ({ label, val, sign='', after=null }) => (
  <div className="border rounded bg-light p-2 text-center h-100">
    <div className="small text-muted">{label}</div>
    <div className="fw-bold fs-5">{sign}{val?.toFixed(2)||'—'}{after}</div>
  </div>
);
const oiColor = str =>
  str?.includes('Bullish') ? 'success' :
  str?.includes('Bearish') ? 'danger' : 'warning';

export default Summary;