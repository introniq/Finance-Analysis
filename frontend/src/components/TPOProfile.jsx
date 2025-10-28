// TPOProfile.jsx – vertical TPO × Price bar chart only
import React, { useMemo, useState } from 'react';
import {
  Card, Row, Col, Container, Badge, Table, Pagination
} from 'react-bootstrap';
import Plot from 'react-plotly.js';
import { motion } from 'framer-motion';

const ITEMS_PER_PAGE = 22;

const TPOProfile = ({ data }) => {
  const [page, setPage] = useState(1);

  /* ----------  single destructuring  ---------- */
  const {
    tpoc, total_tpo, top3_pct, date_range, va_high, va_low, va_diff, va_tpo_pct, supply_check, data: tpoData = []
  } = data || {};

  /* ----------  vertical bar chart  ---------- */
  const chartFigure = useMemo(() => {
    if (!tpoData.length) return { data: [], layout: {} };

    const prices = tpoData.map((r) => parseFloat(r['Price Level']));
    const tpos   = tpoData.map((r) => parseFloat(r['TPO Count']));
    const supply = tpoData.map((r) => r['Supply Check'] || '');

    const colours = supply.map((s) =>
      s.includes('Heavy') ? '#d62728' : s.includes('Demand') ? '#2ca02c' : '#1f77b4'
    );

    const bars = {
      x: prices,
      y: tpos,
      type: 'bar',
      marker: { color: colours, line: { color: 'rgba(0,0,0,0.3)', width: 1 } },
      hovertemplate: '₹%{x:.2f}<br>TPO: %{y:,}<br>Supply: %{customdata}<extra></extra>',
      customdata: supply,
      name: 'TPO',
    };

    const shapes = [];
    const annotations = [];

    // POC line
    if (tpoc && !isNaN(tpoc)) {
      shapes.push({
        type: 'line', x0: tpoc, x1: tpoc, y0: 0, y1: Math.max(...tpos) * 1.05,
        line: { color: '#ff7f0e', width: 3 },
      });
      annotations.push({
        x: tpoc, y: Math.max(...tpos) * 1.06, text: `TPOC ₹${Number(tpoc).toFixed(2)}`,
        showarrow: false, font: { color: '#ff7f0e', size: 11 },
      });
    }

    // VA rect
    if (va_high && va_low && !isNaN(va_high) && !isNaN(va_low)) {
      shapes.push({
        type: 'rect', x0: va_low, x1: va_high, y0: 0, y1: Math.max(...tpos) * 1.05,
        fillcolor: 'rgba(0,176,246,0.15)', line: { width: 0 },
      });
    }

    const layout = {
      title: `TPO Profile – ${date_range || 'Full Range'}`,
      xaxis: { title: 'Price Level (₹)', tickformat: '.2f' },
      yaxis: { title: 'TPO Count (time)', tickformat: ',d' },
      margin: { l: 60, r: 40, t: 40, b: 60 },
      hovermode: 'x unified', shapes, annotations, showlegend: false,
    };

    return { data: [bars], layout };
  }, [tpoData, tpoc, va_high, va_low, date_range]);

  /* ----------  early exit  ---------- */
  if (!data) return (
    <Container fluid className="d-flex align-items-center justify-content-center vh-100">
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-muted text-center">
        <div className="spinner-border text-primary mb-2" role="status" style={{ width: '1.5rem', height: '1.5rem' }} />
        <h6 className="mb-0">No TPO data</h6>
      </motion.div>
    </Container>
  );

  /* ----------  rest of UI  ---------- */
  const supplyColor =
    supply_check?.supply_check?.includes('Demand') ? 'success'
      : supply_check?.supply_check?.includes('Supply') ? 'danger' : 'warning';

  const pages = Math.ceil((tpoData.length || 0) / ITEMS_PER_PAGE);
  const start = (page - 1) * ITEMS_PER_PAGE;
  const visibleRows = tpoData.slice(start, start + ITEMS_PER_PAGE);

  const Tile = ({ icon, label, value, color = 'light' }) => (
    <Card bg={color} text={color === 'dark' ? 'white' : 'dark'} className="shadow-sm h-100 border-0">
      <Card.Body className="d-flex flex-column align-items-center justify-content-center p-2">
        <div className="fs-5 mb-1">{icon}</div>
        <div className="small text-muted text-center">{label}</div>
        <div className="fw-bold fs-6 mt-1">{value}</div>
      </Card.Body>
    </Card>
  );

  const Kpi = ({ label, children }) => (
    <div className="border rounded bg-light px-2 py-1 text-center">
      <div className="small text-muted">{label}</div>
      <div className="fw-bold fs-6">{children}</div>
    </div>
  );

  return (
    <Container fluid className="p-2 bg-light d-flex flex-column vh-100">
      <div className="flex-grow-1 overflow-auto" style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}>
        <style>{`
          .hide-scroll::-webkit-scrollbar { display: none; }
          .hide-scroll { scrollbar-width: none; -ms-overflow-style: none; }
        `}</style>
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
                <Card.Header className="fw-bold bg-primary text-white py-1 px-2 small">TPO Profile Chart</Card.Header>
                <Card.Body className="p-1">
                  {chartFigure.data.length ? (
                    <div className="w-100 hide-scroll" style={{ height: '400px' }}>
                      <Plot
                        data={chartFigure.data}
                        layout={chartFigure.layout}
                        style={{ width: '100%', height: '100%' }}
                        config={{ displayModeBar: true, displaylogo: false }}
                      />
                    </div>
                  ) : (
                    <div className="d-flex align-items-center justify-content-center bg-light text-muted" style={{ height: 320 }}>
                      <i className="fa fa-clock fa-2x" />
                      <span className="ms-2">No TPO data to plot</span>
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
                  <div className="w-100 hide-scroll" style={{ maxHeight: 240 }}>
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
                            <td className="text-center py-0">
                              <Badge bg="info" className="small">{(item['Percentage (%)'] || 0).toFixed(1)}%</Badge>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                  </div>

                  {pages > 1 && (
                    <Pagination size="sm" className="mt-2 mb-1 justify-content-center">
                      <Pagination.Prev onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={page === 1} />
                      <Pagination.Item active>{page}</Pagination.Item>
                      <Pagination.Next onClick={() => setPage((p) => Math.min(pages, p + 1))} disabled={page === pages} />
                    </Pagination>
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

export default TPOProfile;