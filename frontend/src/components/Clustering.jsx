// Clustering.jsx – dashboard-grade, symmetric, compact, nothing hidden
import { Card, Row, Col, Container, Alert } from 'react-bootstrap';
import Plot from 'react-plotly.js';
import { motion } from 'framer-motion';

const Clustering = ({ plot }) => {
  /* ----------  early-exit skeleton  ---------- */
  if (!plot || !Object.keys(plot).length)
    return (
      <Container fluid className="d-flex align-items-center justify-content-center vh-100">
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-muted text-center">
          <div className="spinner-border text-primary mb-2" role="status" style={{ width: '1.5rem', height: '1.5rem' }} />
          <h6 className="mb-0">No clustering data</h6>
        </motion.div>
      </Container>
    );

  /* ----------  tiny KPI helper  ---------- */
  const Tile = ({ icon, label, value, color = 'light' }) => (
    <Card bg={color} text={color === 'dark' ? 'white' : 'dark'} className="shadow-sm h-100 border-0">
      <Card.Body className="d-flex flex-column align-items-center justify-content-center p-2">
        <div className="fs-5 mb-1">{icon}</div>
        <div className="small text-muted text-center">{label}</div>
        <div className="fw-bold fs-6 mt-1">{value}</div>
      </Card.Body>
    </Card>
  );

  /* ----------  render  ---------- */
  return (
    <Container fluid className="p-2 bg-light d-flex flex-column">
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }}>

        {/* KPI row */}
        <Row className="g-2 mb-2">
          <Col xs={6} md={3}><Tile icon="🧩" label="Clusters" value={plot?.data?.length || 0} color="info" /></Col>
          <Col xs={6} md={3}><Tile icon="📊" label="Patterns" value={plot?.data?.[0]?.x?.length || 0} /></Col>
          <Col xs={6} md={3}><Tile icon="🔍" label="Silhouette" value={plot?.layout?.title?.text?.match(/[\d.]+/)?.[0] || '—'} color="warning" /></Col>
          <Col xs={6} md={3}><Tile icon="📅" label="Updated" value="Live" color="dark" /></Col>
        </Row>

        {/* plot card */}
        <Row>
          <Col>
            <Card className="shadow-sm border-0">
              <Card.Header className="fw-bold bg-primary text-white py-1 px-2 small">Pattern Clustering</Card.Header>
              <Card.Body className="p-1">
                <div style={{ width: '100%', height: '100%' }}>
                  <Plot
                    data={plot.data}
                    layout={{
                      ...plot.layout,
                      autosize: true,
                      width: undefined,
                      height: '100%', // fixed height to fit card
                      margin: { l: 50, r: 50, t: 40, b: 50 }
                    }}
                    style={{ width: '100%', height: '100%' }}
                    useResizeHandler={true}
                    config={{ displayModeBar: false }}
                  />
                </div>
              </Card.Body>
            </Card>
          </Col>
        </Row>

        {/* insights alert */}
        <Row className="mt-2">
          <Col>
            <Alert variant="info" className="py-2 px-2 mb-0 small">
              <i className="fas fa-info-circle me-1" />
              <strong>Insight:</strong> Each cluster groups similar market patterns (return, volume, OI). Hover for details.
            </Alert>
          </Col>
        </Row>
      </motion.div>
    </Container>
  );
};

export default Clustering;
