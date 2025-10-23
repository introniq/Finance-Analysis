// Technical.jsx – fully visible, auto-height adaptive
import React from 'react';
import { Card, Row, Col, Container, Pagination } from 'react-bootstrap';
import Plot from 'react-plotly.js';
import { motion } from 'framer-motion';

const ITEMS_PER_PAGE = 3;

const Technical = ({ summary, plots }) => {
  const [page, setPage] = React.useState(1);

  if (!summary || !plots)
    return (
      <Container fluid className="d-flex align-items-center justify-content-center vh-100">
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-muted text-center">
          <div className="spinner-border text-primary mb-2" role="status" />
          <h6>No technical data available</h6>
        </motion.div>
      </Container>
    );

  const plotKeys = ['rsi', 'macd', 'bb', 'stoch', 'adx', 'vwap'];
  const pages = Math.ceil(plotKeys.length / ITEMS_PER_PAGE);
  const visibleKeys = plotKeys.slice((page - 1) * ITEMS_PER_PAGE, page * ITEMS_PER_PAGE);

  /* ---------- KPI Tile ---------- */
  const Tile = ({ icon, label, value, color = 'light' }) => (
    <motion.div whileHover={{ scale: 1.03 }}>
      <Card
        bg={color}
        text={color === 'dark' ? 'white' : 'dark'}
        className="shadow border-0 rounded-4 h-100"
      >
        <Card.Body className="d-flex flex-column align-items-center justify-content-center p-3">
          <div className="fs-4 mb-1">{icon}</div>
          <div className="small text-muted">{label}</div>
          <div className="fw-bold fs-5 mt-1">{value}</div>
        </Card.Body>
      </Card>
    </motion.div>
  );

  return (
    <Container
      fluid
      className="p-3"
      style={{
        background: 'linear-gradient(180deg, #f8f9ff, #ffffff)',
        minHeight: '100vh',
      }}
    >
      <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        {/* KPI SECTION */}
        <Row className="g-3 mb-3">
          <Col xs={6} md={3}>
            <Tile icon="📊" label="Indicators" value={plotKeys.length} color="info" />
          </Col>
          <Col xs={6} md={3}>
            <Tile icon="🎯" label="Page" value={`${page}/${pages}`} />
          </Col>
          <Col xs={6} md={3}>
            <Tile icon="📅" label="Updated" value="Live" color="warning" />
          </Col>
          <Col xs={6} md={3}>
            <Tile icon="⚡" label="Status" value="Active" color="success" />
          </Col>
        </Row>

        {/* INDICATOR CARDS */}
        <div className="overflow-auto" style={{ maxHeight: '78vh', paddingRight: '0.5rem' }}>
          <Row className="g-3">
            {visibleKeys.map((key) => {
              const idx = plotKeys.indexOf(key);
              const item = summary[idx];
              const plotData = plots[key];

              return (
                <Col md={6} lg={4} key={key}>
                  <motion.div whileHover={{ scale: 1.01 }}>
                    <Card
                      className="shadow-lg border-0 rounded-4"
                      style={{
                        background: '#fff',
                        display: 'flex',
                        flexDirection: 'column',
                        height: 'auto',
                      }}
                    >
                      {/* Header */}
                      <Card.Header
                        className="fw-bold text-white small py-2 px-3"
                        style={{
                          background: 'linear-gradient(90deg, #667eea, #764ba2)',
                        }}
                      >
                        {item?.title || key.toUpperCase()}
                      </Card.Header>

                      {/* Body */}
                      <Card.Body className="p-3 d-flex flex-column bg-white">
                        {/* Description */}
                        {item?.text && (
                          <div
                            className="mb-3"
                            style={{
                              overflowY: item.text.length > 5 ? 'auto' : 'visible',
                              maxHeight: '130px',
                            }}
                          >
                            {item.text.map((line, i) => (
                              <p key={i} className="small text-muted mb-1">
                                {line}
                              </p>
                            ))}
                          </div>
                        )}

                        {/* Chart */}
                        <div
                          className="border rounded bg-light p-2"
                          style={{
                            minHeight: '300px',
                            overflowX: 'auto',
                            overflowY: 'hidden',
                          }}
                        >
                          {plotData && Object.keys(plotData).length ? (
                            <div style={{ width: '720px', height: '350px' }}>
                              <Plot
                                data={plotData.data}
                                layout={{
                                  ...plotData.layout,
                                  width: 700,
                                  height: 340,
                                  margin: { l: 50, r: 30, t: 30, b: 40 },
                                  paper_bgcolor: '#ffffff',
                                  plot_bgcolor: '#ffffff',
                                  font: { size: 11, color: '#222' },
                                }}
                                config={{ displayModeBar: false, responsive: true }}
                                style={{ width: '100%', height: '100%' }}
                              />
                            </div>
                          ) : (
                            <div className="d-flex justify-content-center align-items-center text-muted" style={{ height: '300px' }}>
                              <i className="fa fa-chart-line fa-2x" />
                              <small className="ms-2">No Chart Data</small>
                            </div>
                          )}
                        </div>
                      </Card.Body>
                    </Card>
                  </motion.div>
                </Col>
              );
            })}
          </Row>
        </div>

        {/* PAGINATION */}
        {pages > 1 && (
          <Row className="mt-3">
            <Col>
              <Pagination size="sm" className="justify-content-center">
                <Pagination.First onClick={() => setPage(1)} disabled={page === 1} />
                <Pagination.Prev onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={page === 1} />
                {[...Array(pages)].map((_, i) => (
                  <Pagination.Item key={i + 1} active={i + 1 === page} onClick={() => setPage(i + 1)}>
                    {i + 1}
                  </Pagination.Item>
                ))}
                <Pagination.Next onClick={() => setPage((p) => Math.min(pages, p + 1))} disabled={page === pages} />
                <Pagination.Last onClick={() => setPage(pages)} disabled={page === pages} />
              </Pagination>
            </Col>
          </Row>
        )}
      </motion.div>
    </Container>
  );
};

export default Technical;
