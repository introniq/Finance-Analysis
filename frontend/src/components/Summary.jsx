/* eslint-disable react/prop-types */
import React from 'react';
import {
  ListGroup, Container, Row, Col, Card, Badge, Alert
} from 'react-bootstrap';
import { motion } from 'framer-motion';

/*-----------------------------------------------------------*/
/*  100 % defensive – will never throw                     */
/*-----------------------------------------------------------*/
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

  /* ----------  safe destructuring  ---------- */
  const {
    metrics = {},
    delivery_check = {},
    outlook = {},
    pattern = {},
    master_analysis
  } = data;

  const price = live?.price ?? metrics.current_price;

  /* ----------  tiny metric tile  ---------- */
  const MetricTile = ({ icon, label, value, color = 'light' }) => (
    <Card className={`shadow-sm h-100 border-0 bg-${color}`}>
      <Card.Body className="d-flex flex-column align-items-center justify-content-center p-3">
        <div className="fs-2 mb-2">{icon}</div>
        <div className="fw-semibold small text-center">{label}</div>
        <div className="fw-bold fs-5 mt-1">{value ?? '—'}</div>
      </Card.Body>
    </Card>
  );

  /* ----------  master-file display  ---------- */
  const MasterCard = ({ ma }) => {
    if (!ma) return null;

    const stats = ma.summary_stats ?? {};
    const detail = ma.detailed_analysis ?? {};

    const condMeta = {
      delivery_times_3pct:   { icon: '📊', title: 'Delivery Times > 3 %', color: 'warning' },
      amount_1pct_mc:        { icon: '💰', title: 'Amount > 1 % MC',      color: 'danger'  },
      accumulation_2pct_mc:  { icon: '📈', title: 'Accumulation > 2 % MC',color: 'success' }
    };

    return (
      <Card className="shadow-lg border-0 mb-4" style={{ background: 'linear-gradient(145deg,#fff8e1,#ffffff)' }}>
        <Card.Header className="bg-warning text-dark fw-bold">🎯 Master File Analysis Results</Card.Header>
        <Card.Body className="p-4">
          <Row className="mb-3">
            <Col md={6}>
              <h6 className="fw-bold text-primary">Summary</h6>
              <ListGroup variant="flush">
                <ListGroup.Item className="d-flex justify-content-between">
                  <span>Total Conditions Met:</span>
                  <Badge bg="primary">{stats.total_conditions_met ?? 0}</Badge>
                </ListGroup.Item>
                <ListGroup.Item className="d-flex justify-content-between">
                  <span>Qualifying Dates:</span>
                  <Badge bg="info">{stats.qualifying_dates_count ?? 0}</Badge>
                </ListGroup.Item>
                <ListGroup.Item className="d-flex justify-content-between">
                  <span>Market Cap (Cr):</span>
                  <span className="fw-bold">₹{Number(ma.market_cap_cr || 0).toFixed(2)}</span>
                </ListGroup.Item>
              </ListGroup>
            </Col>
            <Col md={6}>
              <h6 className="fw-bold text-primary">Break-down</h6>
              <ListGroup variant="flush">
                {Object.keys(condMeta).map(k => (
                  <ListGroup.Item key={k} className="d-flex justify-content-between">
                    <span>
                      <i className={`me-2 text-${condMeta[k].color} fa fa-${k.includes('delivery')?'truck':k.includes('amount')?'coins':'chart-line'}`} />
                      {condMeta[k].title}
                    </span>
                    <Badge bg={condMeta[k].color}>{stats[`${k.split('_')[0]}_count`] ?? 0}</Badge>
                  </ListGroup.Item>
                ))}
              </ListGroup>
            </Col>
          </Row>

          {/* detail cards */}
          {Object.entries(detail).map(([k, v]) => {
                const meta = condMeta[k.split('_').slice(0,2).join('_')];
                if (!meta) return null;
                return (
                  <Card key={k} className="mb-3 border-0 shadow-sm">
                    <Card.Header className={`bg-${meta.color} text-white py-2`}>
                      <span className="me-2">{meta.icon}</span>
                      <span className="fw-bold">{meta.title}</span>
                      <Badge bg="light" text="dark" className="ms-auto">{v.total_occurrences ?? 0}</Badge>
                    </Card.Header>
                    <Card.Body className="p-3">
                      <p className="small text-muted mb-2">{v.description}</p>
                      {v.results?.length
                        ? (
                          <div style={{ maxHeight: 180, overflowY: 'auto' }}>
                            {v.results.map((r, i) => (
                              <div key={i} className="border rounded p-2 mb-1 bg-white">
                                <div className="d-flex justify-content-between">
                                  <span className="small fw-bold">
                                    {r.start_date ? `${r.start_date} → ${r.end_date}` : r.date}
                                  </span>
                                  <Badge bg={meta.color}>
                                    {r.sum_delivery_times ?? r.percentage_of_mc ?? r.amount ?? '-'}
                                  </Badge>
                                </div>
                                {r.percentage_of_mc && <small className="text-muted">{r.percentage_of_mc}% of MC</small>}
                              </div>
                            ))}
                          </div>
                        )
                        : <p className="text-muted text-center py-2">No qualifying periods</p>}
                    </Card.Body>
                  </Card>
                );
              })}

          {/* qualifying dates */}
          {(ma.qualifying_dates ?? []).length > 0 && (
            <Alert variant="success" className="mt-3 mb-0">
              <Alert.Heading><i className="fas fa-calendar-check me-2" />Qualifying Dates</Alert.Heading>
              <div className="d-flex flex-wrap gap-1">
                {ma.qualifying_dates.slice(0, 12).map(d => <Badge bg="success" key={d}>{d}</Badge>)}
                {ma.qualifying_dates.length > 12 && <Badge bg="secondary">+{ma.qualifying_dates.length - 12} more</Badge>}
              </div>
            </Alert>
          )}
        </Card.Body>
      </Card>
    );
  };

  /* ----------  main render  ---------- */
  return (
    <Container fluid className="px-3">
      {/* MASTER SECTION (first when present) */}
      {master_analysis && (
        <Row className="mb-4">
          <Col><MasterCard ma={master_analysis} /></Col>
        </Row>
      )}

      {/* KPI ROW */}
      <Row className="g-3 mb-4">
        <Col xs={6} md={2}><MetricTile icon="🏢" label="Symbol" value={metrics.symbol} /></Col>
        <Col xs={6} md={3}><MetricTile icon="👥" label="Shares Outstanding" value={metrics.outstanding_shares ? Number(metrics.outstanding_shares).toLocaleString() : '—'} /></Col>
        <Col xs={6} md={3}><MetricTile icon="💰" label="Current Price" value={price ? `₹${Number(price).toFixed(2)}` : '—'} /></Col>
        <Col xs={6} md={2}><MetricTile icon="🏛️" label="Market-Cap (Cr)" value={metrics.market_cap_cr ? `₹${Number(metrics.market_cap_cr).toFixed(2)}` : '—'} /></Col>
        <Col xs={6} md={2}><MetricTile icon="📊" label="52-Wk Range" value={metrics['52w_high_low'] || '—'}  /></Col>
      </Row>

      {/* DELIVERY ALERT */}
      <Alert variant={delivery_check.color || 'secondary'} className="d-flex align-items-center mb-4 py-2">
        <i className={`me-2 fs-5 fa fa-${delivery_check.color === 'success' ? 'check-circle' : 'exclamation-triangle'}`} />
        <div>
          <span className="fw-bold d-block">{delivery_check.message || 'N/A'}</span>
          {delivery_check.delivery_status && <small className="d-block text-muted">{delivery_check.delivery_status}</small>}
          {delivery_check.oi_status && <small className="d-block text-muted">{delivery_check.oi_status}</small>}
        </div>
      </Alert>

      {/* OUTLOOK + PATTERN COLUMNS */}
      <Row className="g-4">
        <Col lg={6} className="d-flex">
          <Card className="shadow w-100 border-0">
            <Card.Header className="bg-primary text-white fw-bold">Market Outlook</Card.Header>
            <Card.Body className="p-3 d-flex flex-column justify-content-between">
              <Row className="g-3 mb-3">
                <Col xs={6}><Kpi label="Latest Close" val={outlook.close} sign="₹" /></Col>
                <Col xs={6}><Kpi label="5-Day MA" val={outlook.ma5} sign="₹" /></Col>
                <Col xs={6}><Kpi label="PCR (curr)" val={outlook.pcr} after={live?.pcr_update && <Badge bg="success" className="ms-2">{Number(live.pcr_update).toFixed(2)}</Badge>} /></Col>
                <Col xs={6}><Kpi label="PCR 5-D Avg" val={outlook.pcr_ma5} /></Col>
              </Row>
              <div className="text-center mt-auto">
                <h5 className={`fw-bold mb-2 ${outlook.wyckoff_outlook === 'Bullish' ? 'text-success' : outlook.wyckoff_outlook === 'Bearish' ? 'text-danger' : 'text-warning'}`}>
                  <i className={`fa fa-arrow-${outlook.wyckoff_outlook === 'Bullish' ? 'up' : outlook.wyckoff_outlook === 'Bearish' ? 'down' : 'minus'} me-2`} />
                  {outlook.wyckoff_event || 'Neutral'} – {outlook.wyckoff_outlook || 'Neutral'} bias
                </h5>
                <Badge bg={oiColor(outlook.oi_divergence)} className="px-3 py-2">{outlook.oi_divergence || 'No divergence'}</Badge>
              </div>
            </Card.Body>
          </Card>
        </Col>

        <Col lg={6} className="d-flex">
          <Card className="shadow w-100 border-0">
            <Card.Header className="bg-secondary text-white fw-bold">Pattern & Guidance</Card.Header>
            <Card.Body className="p-3 d-flex flex-column justify-content-between">
              <ListGroup variant="flush" className="border-0">
                <ListGroup.Item className="d-flex justify-content-between px-0">
                  <span><i className="fa fa-list text-info me-2" />Recent FOI (last 10)</span>
                  <Badge bg="info" className="px-2">{pattern.recent_foi?.length >= 10 ? pattern.recent_foi.map(p => Number(p).toFixed(2)).join(', ') : '—'}</Badge>
                </ListGroup.Item>
                <ListGroup.Item className="d-flex justify-content-between px-0">
                  <span><i className="fa fa-percentage text-warning me-2" />Exp. Return</span>
                  <Badge bg={pattern.expected_change > 0 ? 'success' : 'danger'}>{pattern.expected_change > 0 ? '+' : ''}{Number(pattern.expected_change || 0).toFixed(1)}%</Badge>
                </ListGroup.Item>
                <ListGroup.Item className="d-flex justify-content-between px-0">
                  <span><i className="fa fa-compass text-success me-2" />Guidance</span>
                  <Badge bg={pattern.guidance === 'Bullish' ? 'success' : pattern.guidance === 'Bearish' ? 'danger' : 'warning'}>{pattern.guidance || 'Neutral'}</Badge>
                </ListGroup.Item>
                <ListGroup.Item className="d-flex justify-content-between px-0">
                  <span><i className="fa fa-eye text-secondary me-2" />Wyckoff</span>
                  <Badge bg="dark">{pattern.wyckoff_event || 'Neutral'}</Badge>
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
const Kpi = ({ label, val, sign = '', after = null }) => (
  <div className="border rounded bg-light p-2 text-center h-100">
    <div className="small text-muted">{label}</div>
    <div className="fw-bold fs-5">{sign}{val != null ? Number(val).toFixed(2) : '—'}{after}</div>
  </div>
);
const oiColor = str =>
  str?.includes('Bullish') ? 'success' : str?.includes('Bearish') ? 'danger' : 'warning';

export default Summary;