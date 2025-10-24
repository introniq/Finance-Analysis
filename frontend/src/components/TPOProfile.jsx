// TPOProfile.jsx  –  dashboard-grade, symmetric, fully visible, scroll where needed
import React from 'react';
import { Card, Row, Col, Container, Badge, Table, Pagination } from 'react-bootstrap';
import Plot from 'react-plotly.js';
import { motion } from 'framer-motion';

const ITEMS_PER_PAGE = 10;

const TPOProfile = ({ data }) => {
  const [page, setPage] = React.useState(1);

  /* ----------  early-exit skeleton  ---------- */
  if (!data)
    return (
      <Container fluid className="d-flex align-items-center justify-content-center vh-100">
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-muted text-center">
          <div className="spinner-border text-primary mb-2" role="status" style={{ width: '1.5rem', height: '1.5rem' }} />
          <h6 className="mb-0">No TPO data</h6>
        </motion.div>
      </Container>
    );

  const {
    data: tpoData,
    tpoc,
    total_tpo,
    top3_pct,
    date_range,
    plot,
    va_high,
    va_low,
    va_diff,
    va_tpo_pct,
    supply_check,
  } = data;

  const supplyColor =
    supply_check?.supply_check?.includes('Demand') ? 'success' :
      supply_check?.supply_check?.includes('Supply') ? 'danger' : 'warning';

  /* ----------  pagination  ---------- */
  const pages = Math.ceil((tpoData?.length || 0) / ITEMS_PER_PAGE);
  const start = (page - 1) * ITEMS_PER_PAGE;
  const visibleRows = (tpoData || []).slice(start, start + ITEMS_PER_PAGE);

  /* ----------  reusable metric tile  ---------- */
  const Tile = ({ icon, label, value, color = 'light' }) => (
    <Card bg={color} text={color === 'dark' ? 'white' : 'dark'} className="shadow-sm h-100 border-0">
      <Card.Body className="d-flex flex-column align-items-center justify-content-center p-2">
        <div className="fs-5 mb-1">{icon}</div>
        <div className="small text-muted text-center">{label}</div>
        <div className="fw-bold fs-6 mt-1">{value}</div>
      </Card.Body>
    </Card>
  );

  /* ----------  render  ---------- */
  return (
    <Container fluid className="p-2 bg-light vh-100 d-flex flex-column">
      {/* hide all scrollbars globally for this container */}
      <style>{`
        .hide-scrollbar {
          scrollbar-width: none; /* Firefox */
          -ms-overflow-style: none; /* IE 10+ */
        }
        .hide-scrollbar::-webkit-scrollbar {
          display: none; /* Chrome, Safari, Opera */
        }
      `}</style>

      {/* scrollable viewport ------------------------------------ */}
      <div className="flex-grow-1 overflow-auto hide-scrollbar">
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }}>

          {/* KPI row */}
          <Row className="g-2 mb-2">
            <Col xs={6} md={3}><Tile icon="🎯" label="TPOC" value={`₹${Number(tpoc || 0).toFixed(2)}`} color="info" /></Col>
            <Col xs={6} md={3}><Tile icon="📊" label="Total TPO" value={(total_tpo || 0).toLocaleString()} /></Col>
            <Col xs={6} md={3}><Tile icon="🔝" label="Top 3 %" value={`${Number(top3_pct || 0).toFixed(1)}%`} color="warning" /></Col>
            <Col xs={6} md={3}><Tile icon="📅" label="Range" value={date_range || '—'} color="dark" /></Col>
          </Row>

          {/* value area bar */}
          <Row className="mb-2">
            <Col>
              <Card className="shadow-sm border-0">
                <Card.Body className="d-flex flex-wrap align-items-center justify-content-between gap-2 p-2">
                  <div className="d-flex gap-2">
                    <Kpi label="VA High">₹{Number(va_high || 0).toFixed(2)}</Kpi>
                    <Kpi label="VA Low">₹{Number(va_low || 0).toFixed(2)}</Kpi>
                    <Kpi label="Width">₹{Number(va_diff || 0).toFixed(2)}</Kpi>
                    <Kpi label="VA %">{Number(va_tpo_pct || 0).toFixed(1)}%</Kpi>
                  </div>
                  <Badge bg={supplyColor} className="px-2 py-1 small">{supply_check?.supply_check || 'Balanced'}</Badge>
                </Card.Body>
              </Card>
            </Col>
          </Row>

          {/* plot */}
          <Row className="mb-2">
            <Col>
              <Card className="shadow-sm border-0">
                <Card.Header className="fw-bold bg-primary text-white py-1 px-2 small">
                  TPO Profile Chart
                </Card.Header>
                <Card.Body className="p-1">
                  {plot && Object.keys(plot).length ? (
                    <div
                      className="w-100 overflow-auto hide-scrollbar"
                      style={{ height: 500 }}
                    >
                      <div className="plot-scroll hide-scrollbar" style={{ minWidth: 800, height: '100%' }}>
                        <Plot
                          data={plot.data}
                          layout={{
                            ...plot.layout,
                            autosize: true,
                            margin: { l: 40, r: 20, t: 30, b: 30 },
                          }}
                          style={{ width: '100%', height: '100%' }}
                          useResizeHandler={true}
                          config={{ displayModeBar: false }}
                        />
                      </div>
                    </div>
                  ) : (
                    <div
                      className="d-flex align-items-center justify-content-center bg-light text-muted"
                      style={{ height: 320 }}
                    >
                      <i className="fa fa-clock fa-2x" />
                    </div>
                  )}
                </Card.Body>
              </Card>
            </Col>
          </Row>

          {/* table */}
          <Row>
            <Col>
              <Card className="shadow-sm border-0 h-100 mb-3">
                <Card.Header className="fw-bold bg-secondary text-white py-1 px-2 small">Top TPO Levels</Card.Header>
                <Card.Body className="p-1 d-flex flex-column">
                  <div className="w-100 overflow-auto hide-scrollbar" style={{ maxHeight: 240 }}>
                    <div className="tpo-scroll hide-scrollbar">
                      <Table size="sm" hover responsive className="mb-0">
                        <thead className="table-dark">
                          <tr>
                            <th className="text-center py-1">Price</th>
                            <th className="text-center py-1">TPO</th>
                            <th className="text-center py-1">Share</th>
                          </tr>
                        </thead>
                        <tbody>
                          {visibleRows.map((item, i) => (
                            <tr key={i} style={{ height: 18 }}>
                              <td className="text-center py-0">₹{Number(item['Price Level'] || 0).toFixed(2)}</td>
                              <td className="text-center py-0">{(item['TPO Count'] || 0).toLocaleString()}</td>
                              <td className="text-center py-0"><Badge bg="info" className="small">{(item['Percentage (%)'] || 0).toFixed(1)}%</Badge></td>
                            </tr>
                          ))}
                        </tbody>
                      </Table>
                    </div>
                  </div>

                  {/* Tables */}
                  <Row>
                    <Col>
                      <Card className="shadow-sm border-0 h-100 mb-3">
                        <Card.Header className="fw-bold bg-secondary text-white py-1 px-2 small">Top TPO Levels</Card.Header>
                        <Card.Body className="p-1 d-flex flex-column">
                          <div className="w-100 overflow-auto hide-scrollbar" style={{ maxHeight: 240 }}>
                            <Table size="sm" hover responsive className="mb-0" style={{ tableLayout: 'fixed' }}>
                              <thead className="table-dark">
                                <tr>
                                  <th className="text-center py-1">Price</th>
                                  <th className="text-center py-1">TPO</th>
                                  <th className="text-center py-1">Share</th>
                                </tr>
                              </thead>
                              <tbody>
                                {tpoData.map((item, i) => (
                                  <tr key={i} style={{ height: 18 }}>
                                    <td className="text-center py-0">₹{Number(item['Price Level'] || 0).toFixed(2)}</td>
                                    <td className="text-center py-0">{(item['TPO Count'] || 0).toLocaleString()}</td>
                                    <td className="text-center py-0">
                                      <Badge bg="info" className="small">{(item['Percentage (%)'] || 0).toFixed(1)}%</Badge>
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </Table>
                          </div>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>

                </Card.Body>
              </Card>
            </Col>
          </Row>
        </motion.div>
      </div>
    </Container>
  );
};

/* ----------  tiny helper  ---------- */
const Kpi = ({ label, children }) => (
  <div className="border rounded bg-light px-2 py-1 text-center">
    <div className="small text-muted">{label}</div>
    <div className="fw-bold fs-6">{children}</div>
  </div>
);

export default TPOProfile;
