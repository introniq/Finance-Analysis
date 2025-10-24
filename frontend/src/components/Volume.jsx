// Volume.jsx  –  full-page scroll, compact sizes, NEW user peak-diff picker
import React, { useEffect, useState } from 'react';
import { Card, Row, Col, Container, Badge, Table, Pagination, Form, Button, Alert } from 'react-bootstrap';
import Plot from 'react-plotly.js';
import { motion } from 'framer-motion';

const PAGE_SIZE = 22;
const API_BASE = 'http://localhost:8050';  // FIXED: Add API base URL (matches Fileuploader)

const Volume = ({ file, data: initialData, onData }) => {  // FIXED: Accept initial data from props to avoid re-fetch if available
  console.log('Volume received file:', file);
  console.log('Volume received initial data:', initialData);  // Debug: Check if data is passed
  /* ---------- local states ---------- */
  const [data, setData] = useState(initialData || null);  // FIXED: Use initialData if provided
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [peakStart, setPeakStart] = useState('');
  const [peakEnd, setPeakEnd] = useState('');
  const [page, setPage] = React.useState(1);
  const [hasFetched, setHasFetched] = useState(false);  // NEW: Prevent multiple initial fetches

  /* ---------- fetch data (with optional peak range) ---------- */
  const fetchAnalysis = async (start, end, isInitial = false) => {
    if (!file) {
      console.warn('Volume: No file provided, skipping fetch');
      setError('No file available—please upload again.');
      return;
    }
    if (!isInitial && hasFetched) {  // FIXED: Only allow re-fetch on peak apply, skip if already fetched initially
      console.log('Volume: Skipping fetch - already loaded, using peak params');
    }
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);
    const params = { windowSize: 10, clusters: 3 };
    if (start && end) params.peakDiffDates = [start, end];
    formData.append('params', JSON.stringify(params));

    try {
      // FIXED: Use full API_BASE URL instead of relative '/analyze'
      const res = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: formData });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      const json = await res.json();
      if (json.error) throw new Error(json.error);
      setData(json.volume || null);
      if (onData) onData(json);
      setHasFetched(true);  // NEW: Mark as fetched
    } catch (err) {
      console.error('Volume fetch error:', err);
      setError(`Fetch failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  /* ---------- auto-run once file changes (only if no initial data) ---------- */
  useEffect(() => {
    if (file && !initialData && !hasFetched) {  // FIXED: Skip if initialData provided or already fetched
      fetchAnalysis(peakStart, peakEnd, true);
    }
  }, [file, initialData, hasFetched]);

  /* ---------- user clicked "Apply dates" ---------- */
  const handlePeakApply = () => {
    fetchAnalysis(peakStart, peakEnd, false);  // NEW: Pass flag for non-initial fetch
  };

  /* ---------- render helpers ---------- */
  const {
    data: volumeData,
    poc,
    total_vol,
    top3_pct,
    date_range,
    plot,
    va_high,
    va_low,
    va_diff,
    va_vol_pct,
    supply_check,
    peak_diff,
  } = data || {};

  const supplyColor =
    supply_check?.supply_check?.includes('Demand') ? 'success' :
      supply_check?.supply_check?.includes('Supply') ? 'danger' : 'warning';

  const pages = Math.ceil((volumeData || []).length / PAGE_SIZE);
  const startIdx = (page - 1) * PAGE_SIZE;
  const visibleRows = (volumeData || []).slice(startIdx, startIdx + PAGE_SIZE);

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

  /* ---------- empty / loading / error ---------- */
  if (!file && !initialData) return (
    <Container fluid className="d-flex align-items-center justify-content-center vh-100">
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-muted text-center">
        <div className="spinner-border text-primary mb-2" role="status" style={{ width: '1.5rem', height: '1.5rem' }} />
        <h6 className="mb-0">Waiting for file</h6>
      </motion.div>
    </Container>
  );

  if (error) return (
    <Container fluid className="d-flex align-items-center justify-content-center vh-100">
      <Alert variant="danger" className="w-50">
        <Alert.Heading>Analysis Error</Alert.Heading>
        <p>{error}</p>
        <Button variant="outline-danger" onClick={() => window.location.reload()}>Retry Upload</Button>
      </Alert>
    </Container>
  );

  if (loading && !data) return (  // FIXED: Only show loading if no data yet
    <Container fluid className="d-flex align-items-center justify-content-center vh-100">
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-center">
        <div className="spinner-border text-primary mb-2" role="status" style={{ width: '2rem', height: '2rem' }} />
        <h6 className="mb-0">Analyzing volume profile...</h6>
        <small>Peak-diff calculation in progress</small>
      </motion.div>
    </Container>
  );

  /* ---------- main UI ---------- */
  return (
    <Container fluid className="p-2 bg-light vh-100 d-flex flex-column">
      <div className="flex-grow-1" style={{overflowY: "scroll",
        overflowX: "hidden",
        scrollbarWidth: "none",
        msOverflowStyle: "none",}}>
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }} style={{
          overflowY: "scroll",
          overflowX: "hidden",
          scrollbarWidth: "none",
          msOverflowStyle: "none",
        }}>

          {/* KPI row */}
          <Row className="g-2 mb-2">
            <Col xs={6} md={3}><Tile icon="🎯" label="POC" value={`₹${poc || '-'}`} color="info" /></Col>
            <Col xs={6} md={3}><Tile icon="📊" label="Total Vol" value={total_vol?.toLocaleString() || '-'} /></Col>
            <Col xs={6} md={3}><Tile icon="🔝" label="Top 3 %" value={`${top3_pct?.toFixed(1) || '-'}%`} color="warning" /></Col>
            <Col xs={6} md={3}><Tile icon="📅" label="Range" value={date_range || '-'} color="dark" /></Col>
          </Row>

          {/* Value-area bar + NEW user peak-diff picker */}
          <Row className="mb-2">
            <Col>
              <Card className="shadow-sm border-0">
                <Card.Body className="d-flex flex-wrap align-items-center justify-content-between gap-2 p-2">
                  <div className="d-flex gap-2">
                    <Kpi label="VA High">₹{va_high || '-'}</Kpi>
                    <Kpi label="VA Low">₹{va_low || '-'}</Kpi>
                    <Kpi label="Width">₹{va_diff?.toFixed(2) || '-'}</Kpi>
                    <Kpi label="VA %">{va_vol_pct?.toFixed(1) || '-'}%</Kpi>
                    {peak_diff != null && (
                      <Kpi label="High-Peaks Diff">₹{peak_diff.toFixed(2)}</Kpi>
                    )}
                  </div>
                  <Badge bg={supplyColor} className="px-2 py-1 small">{supply_check?.supply_check || 'N/A'}</Badge>
                </Card.Body>
              </Card>
            </Col>
          </Row>

          {/* User date pickers */}
          <Row className="mb-2">
            <Col>
              <Card className="shadow-sm border-0">
                <Card.Body className="d-flex align-items-center gap-2 p-2">
                  <div className="fw-bold small">High-Peaks diff between</div>
                  <Form.Control type="date" size="sm" style={{ width: 160 }}
                    value={peakStart} onChange={e => setPeakStart(e.target.value)} />
                  <span className="small">and</span>
                  <Form.Control type="date" size="sm" style={{ width: 160 }}
                    value={peakEnd} onChange={e => setPeakEnd(e.target.value)} />
                  <Button size="sm" variant="primary"
                    onClick={handlePeakApply} disabled={!peakStart || !peakEnd || loading}>
                    {loading ? '…' : 'Apply'}
                  </Button>
                </Card.Body>
              </Card>
            </Col>
          </Row>

          {/* Plot */}
          <Row className="mb-2">
            <Col>
              <Card className="shadow-sm border-0">
                <Card.Header className="fw-bold bg-primary text-white py-1 px-2 small">Volume Profile</Card.Header>
                <Card.Body className="p-1">
                  {plot && Object.keys(plot).length ? (
                    <div className="w-100 overflow-auto" >
                      <div style={{ height: '100%' }}>
                        <Plot
                          data={plot.data}
                          layout={{ ...plot.layout, autosize: true, margin: { l: 40, r: 20, t: 50, b: 30 } }}
                          style={{ width: '100%', height: '100%' }}
                          useResizeHandler={true}
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

          {/* Table */}
          <Row>
            <Col>
              <Card className="shadow-sm border-0 h-100 mb-3">
                <Card.Header className="fw-bold bg-secondary text-white py-1 px-2 small">
                  Top Price Levels – with Date & Historical Supply Check
                </Card.Header>
                <Card.Body className="p-1 d-flex flex-column">
                  <div className="w-100 overflow-auto" style={{
                    maxHeight: 300, scrollbarWidth: 'thin', overflowY: "scroll",
                    overflowX: "hidden",
                    scrollbarWidth: "none",
                    msOverflowStyle: "none",
                  }}>

                    <div className="table-scroll">
                      <Table size="sm" hover responsive className="mb-0">
                        <thead className="table-dark">
                          <tr>
                            <th className="text-center py-1">Date</th>
                            <th className="text-center py-1">Price</th>
                            <th className="text-center py-1">Volume</th>
                            <th className="text-center py-1">Share</th>
                            <th className="text-center py-1">Supply Now</th>
                            <th className="text-center py-1">Historical Supply*</th>
                          </tr>
                        </thead>
                        <tbody>
                          {visibleRows.map((row, i) => (
                            <tr key={i}>
                              <td className="text-center py-1">{row.Date ? new Date(row.Date).toLocaleDateString('en-GB') : '-'}</td>
                              <td className="text-center py-1">₹{row['Price Level']}</td>
                              <td className="text-center py-1">{Number(row.Volume).toLocaleString()}</td>
                              <td className="text-center py-1"><Badge bg="info" className="small">{row['Percentage (%)']}%</Badge></td>
                              <td className="text-center py-1">
                                <Badge bg={row.Supply_Check?.includes('Heavy') ? 'danger' : row.Supply_Check?.includes('Demand') ? 'success' : 'warning'} className="small">
                                  {row.Supply_Check || 'N/A'}
                                </Badge>
                              </td>
                              <td className="text-center py-1">
                                <Badge bg={row.Historical_Supply_Count > 0 ? 'secondary' : 'light'} text={row.Historical_Supply_Count > 0 ? 'white' : 'dark'} className="small">
                                  {row.Historical_Supply_Count ?? 0} Heavy
                                </Badge>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </Table>
                    </div>
                  </div>

                  {pages > 1 && (
                    <Pagination size="sm" className="mt-2 mb-1 justify-content-center">
                      <Pagination.Prev onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1} />
                      <Pagination.Item active>{page}</Pagination.Item>
                      <Pagination.Next onClick={() => setPage(p => Math.min(pages, p + 1))} disabled={page === pages} />
                    </Pagination>
                  )}
                  <div className="mt-1 mb-0 small text-muted">
                    *Historical Supply = how many past sessions this bucket was tagged “Heavy Supply”.
                  </div>
                </Card.Body>
              </Card>
            </Col>
          </Row>
        </motion.div>
      </div>
    </Container>
  );
};

export default Volume;