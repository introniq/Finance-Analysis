// OIProfile.jsx  –  dashboard-grade, symmetric, fully visible, scroll where needed
import React from 'react';
import {
  Card, Row, Col, Container, Badge, Table, Pagination
} from 'react-bootstrap';
import Plot from 'react-plotly.js';
import { motion } from 'framer-motion';

const ITEMS_PER_PAGE = 22;

const OIProfile = ({ data }) => {
  const [page, setPage] = React.useState(1);

  /* ----------  early-exit skeleton  ---------- */
  if (!data)
    return (
      <Container fluid className="d-flex align-items-center justify-content-center vh-100">
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-muted text-center">
          <div className="spinner-border text-primary mb-2" role="status" style={{ width: '1.5rem', height: '1.5rem' }} />
          <h6 className="mb-0">No OI data</h6>
        </motion.div>
      </Container>
    );

  /* ----------  destructuring with safe defaults  ---------- */
  const {
    data: oiData,
    poc,
    total_oi,
    top3_pct,
    date_range,
    plot,
    va_high,
    va_low,
    va_diff,
    va_oi_pct,
    supply_check,
    cumulative_data,
    /* ---- new keys ---- */
    edge_diagonal,
    cumulative_open_close,
    turnaround_point,
  } = data;

  const supplyColor =
    supply_check?.supply_check?.includes('Demand') ? 'success' :
    supply_check?.supply_check?.includes('Supply')  ? 'danger'  : 'warning';

  /* ----------  pagination  ---------- */
  const pages = Math.ceil((oiData?.length || 0) / ITEMS_PER_PAGE);
  const start   = (page - 1) * ITEMS_PER_PAGE;
  const visibleRows = (oiData || []).slice(start, start + ITEMS_PER_PAGE);

  /* ----------  reusable metric tile  ---------- */
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
    <Container fluid className="p-2 bg-light vh-100 d-flex flex-column">
      {/* scrollable viewport ------------------------------------ */}
      <div className="flex-grow-1 overflow-auto">
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }}>

          {/* KPI row */}
          <Row className="g-2 mb-2">
            <Col xs={6} md={3}><Tile icon="🎯" label="POC" value={`₹${Number(poc || 0).toFixed(2)}`} color="info" /></Col>
            <Col xs={6} md={3}><Tile icon="📊" label="Total OI" value={(total_oi || 0).toLocaleString()} /></Col>
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
                    <Kpi label="VA OI %">{Number(va_oi_pct || 0).toFixed(1)}%</Kpi>
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
                <Card.Header className="fw-bold bg-primary text-white py-1 px-2 small">OI Profile Chart</Card.Header>
                <Card.Body className="p-1">
                  {plot && Object.keys(plot).length ? (
                    <div className="w-100 overflow-auto" style={{ height: 320 }}>
                      <div style={{ width: 1200, height: 700 }}>
                        <Plot
                          data={plot.data}
                          layout={{ ...plot.layout, width: 1200, height: 700, margin: { l: 40, r: 20, t: 10, b: 30 } }}
                          config={{ displayModeBar: false }}
                        />
                      </div>
                    </div>
                  ) : (
                    <div className="d-flex align-items-center justify-content-center bg-light text-muted" style={{ height: 320 }}>
                      <i className="fa fa-bar-chart fa-2x" />
                    </div>
                  )}
                </Card.Body>
              </Card>
            </Col>
          </Row>

          {/* ----------  NEW : Edge / Diagonal OI  ---------- */}
          {edge_diagonal && (
            <Row className="mb-2">
              <Col>
                <Card className="shadow-sm border-0">
                  <Card.Header className="fw-bold bg-info text-white py-1 px-2 small">Edge / Diagonal OI</Card.Header>
                  <Card.Body className="p-2 d-flex flex-wrap gap-3">
                    <Kpi label="Edge OI">{(edge_diagonal.edge_oi || 0).toLocaleString()}</Kpi>
                    <Kpi label="Edge %">{Number(edge_diagonal.edge_pct || 0).toFixed(1)}%</Kpi>
                    <Kpi label="Diagonal OI">{(edge_diagonal.diagonal_oi || 0).toLocaleString()}</Kpi>
                    <Kpi label="Diagonal %">{Number(edge_diagonal.diagonal_pct || 0).toFixed(1)}%</Kpi>
                  </Card.Body>
                </Card>
              </Col>
            </Row>
          )}

          {/* ----------  NEW : Turn-around Point  ---------- */}
          {turnaround_point && (
            <Row className="mb-2">
              <Col>
                <Card className="shadow-sm border-0">
                  <Card.Header className="fw-bold bg-dark text-white py-1 px-2 small">OI Turn-Around Point</Card.Header>
                  <Card.Body className="p-2 d-flex flex-wrap gap-3">
                    <Kpi label="Turn Price">₹{Number(turnaround_point.turnaround_price || 0).toFixed(2)}</Kpi>
                    <Kpi label="Cum %">{Number(turnaround_point.turnaround_pct || 0).toFixed(1)}%</Kpi>
                  </Card.Body>
                </Card>
              </Col>
            </Row>
          )}

          {/* ----------  NEW : Cumulative OI (Open-to-Close)  ---------- */}
          {cumulative_open_close && cumulative_open_close.length > 0 && (
            <Row className="mb-2">
              <Col>
                <Card className="shadow-sm border-0">
                  <Card.Header className="fw-bold bg-secondary text-white py-1 px-2 small">Cumulative OI (Open-to-Close)</Card.Header>
                  <Card.Body className="p-1">
                    <div className="w-100 overflow-auto" style={{ maxHeight: 200 }}>
                      <Table size="sm" hover responsive className="mb-0">
                        <thead className="table-dark">
                          <tr>
                            <th className="text-center py-1">Price</th>
                            <th className="text-center py-1">Cum OI Up</th>
                            <th className="text-center py-1">Cum OI Down</th>
                          </tr>
                        </thead>
                        <tbody>
                          {cumulative_open_close.map((row, i) => (
                            <tr key={i} style={{ height: 20 }}>
                              <td className="text-center py-0">₹{Number(row['Price Level'] || 0).toFixed(2)}</td>
                              <td className="text-center py-0">{(row['Cum_OI_Up'] || 0).toLocaleString()}</td>
                              <td className="text-center py-0">{(row['Cum_OI_Down'] || 0).toLocaleString()}</td>
                            </tr>
                          ))}
                        </tbody>
                      </Table>
                    </div>
                  </Card.Body>
                </Card>
              </Col>
            </Row>
          )}

          {/* table */}
          <Row>
            <Col>
              <Card className="shadow-sm border-0 h-100 mb-3">
                <Card.Header className="fw-bold bg-secondary text-white py-1 px-2 small">Top OI Levels</Card.Header>
                <Card.Body className="p-1 d-flex flex-column">
                  <div className="w-100 overflow-auto" style={{ maxHeight: 240, scrollbarWidth: 'auto' }}>
                    <style>{`
                      .oi-table-scroll::-webkit-scrollbar{width:8px;height:8px}
                      .oi-table-scroll::-webkit-scrollbar-thumb{background:#888;border-radius:4px}
                      .oi-table-scroll::-webkit-scrollbar-track{background:#f1f1f1}
                    `}</style>
                    <div className="oi-table-scroll">
                      <Table size="sm" hover responsive className="mb-0">
                        <thead className="table-dark">
                          <tr>
                            <th className="text-center py-1">Price</th>
                            <th className="text-center py-1">OI</th>
                            <th className="text-center py-1">Share</th>
                          </tr>
                        </thead>
                        <tbody>
                          {visibleRows.map((item, i) => (
                            <tr key={i} style={{ height: 20 }}>
                              <td className="text-center py-0">₹{Number(item['Price Level'] || 0).toFixed(2)}</td>
                              <td className="text-center py-0">{(item.OI || 0).toLocaleString()}</td>
                              <td className="text-center py-0"><Badge bg="info" className="small">{(item['Percentage (%)'] || 0).toFixed(1)}%</Badge></td>
                            </tr>
                          ))}
                        </tbody>
                      </Table>
                    </div>
                  </div>

                  {pages > 1 && (
                    <Pagination size="sm" className="mt-2 mb-1 justify-content-center">
                      <Pagination.First onClick={() => setPage(1)} disabled={page === 1} />
                      <Pagination.Prev  onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={page === 1} />
                      {[...Array(Math.min(5, pages))].map((_, i) => {
                        const p = Math.max(1, Math.min(pages - 4, page - 2)) + i;
                        return (
                          <Pagination.Item key={p} active={p === page} onClick={() => setPage(p)}>
                            {p}
                          </Pagination.Item>
                        );
                      })}
                      <Pagination.Next onClick={() => setPage((p) => Math.min(pages, p + 1))} disabled={page === pages} />
                      <Pagination.Last  onClick={() => setPage(pages)} disabled={page === pages} />
                    </Pagination>
                  )}

                  {cumulative_data && cumulative_data.length > 0 && (
                    <div className="mt-2 p-2 bg-light rounded small">
                      <strong>Cumulative:</strong> Bottom-Up {(cumulative_data[cumulative_data.length - 1]?.Cum_Pct_Bottom_Up || 0).toFixed(1)}% |
                      Top-Down {(cumulative_data[cumulative_data.length - 1]?.Cum_Pct_Top_Down || 0).toFixed(1)}%
                    </div>
                  )}
                </Card.Body>
              </Card>
            </Col>
          </Row>
        </motion.div>
      </div>
    </Container>
  );
};

/* ----------  tiny helper  ---------- */
const Kpi = ({ label, children }) => (
  <div className="border rounded bg-light px-2 py-1 text-center">
    <div className="small text-muted">{label}</div>
    <div className="fw-bold fs-6">{children}</div>
  </div>
);

export default OIProfile;