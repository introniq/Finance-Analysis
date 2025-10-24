// Technical.jsx – fully visible, auto-height adaptive, fullscreen button on header
import React, { useState } from 'react';
import { Card, Row, Col, Container, Button, Modal } from 'react-bootstrap';
import Plot from 'react-plotly.js';
import { motion } from 'framer-motion';
import { FaExpand } from 'react-icons/fa'; // fullscreen icon

const Technical = ({ summary, plots }) => {
  const [modalData, setModalData] = useState(null);

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

  const Tile = ({ icon, label, value, color = 'light' }) => (
    <motion.div whileHover={{ scale: 1.03 }}>
      <Card bg={color} text={color === 'dark' ? 'white' : 'dark'} className="shadow border-0 rounded-4 h-100">
        <Card.Body className="d-flex flex-column align-items-center justify-content-center p-3">
          <div className="fs-4 mb-1">{icon}</div>
          <div className="small text-muted">{label}</div>
          <div className="fw-bold fs-5 mt-1">{value}</div>
        </Card.Body>
      </Card>
    </motion.div>
  );

  return (
    <Container fluid className="p-3" style={{ background: 'linear-gradient(180deg, #f8f9ff, #ffffff)', minHeight: '100vh' }}>
      <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        {/* KPI SECTION */}
        <Row className="g-3 mb-3">
          <Col xs={6} md={3}><Tile icon="📊" label="Indicators" value={plotKeys.length} color="info" /></Col>
          <Col xs={6} md={3}><Tile icon="📅" label="Updated" value="Live" color="warning" /></Col>
          <Col xs={6} md={3}><Tile icon="⚡" label="Status" value="Active" color="success" /></Col>
        </Row>

        {/* INDICATOR CARDS */}
        <div className="overflow-auto" style={{ maxHeight: '78vh', paddingRight: '0.5rem', scrollbarWidth: 'none', msOverflowStyle: 'none' }}>
          <style>{`.overflow-auto::-webkit-scrollbar { display: none; }`}</style>
          <Row className="g-3">
            {plotKeys.map((key) => {
              const idx = plotKeys.indexOf(key);
              const item = summary[idx];
              const plotData = plots[key];

              return (
                <Col md={6} lg={4} key={key}>
                  <motion.div whileHover={{ scale: 1.01 }}>
                    <Card className="shadow-lg border-0 rounded-4" style={{ background: '#fff', display: 'flex', flexDirection: 'column', height: 'auto' }}>
                      {/* Header with fullscreen button */}
                      <Card.Header className="d-flex justify-content-between align-items-center small py-2 px-3" style={{ background: 'linear-gradient(90deg, #667eea, #764ba2)', color: 'white' }}>
                        <span>{item?.title || key.toUpperCase()}</span>
                        {plotData && Object.keys(plotData).length > 0 && (
                          <Button variant="light" size="sm" onClick={() => setModalData(plotData)} style={{ padding: '0.25rem 0.5rem' }}>
                            <FaExpand />
                          </Button>
                        )}
                      </Card.Header>

                      {/* Body */}
                      <Card.Body className="p-3 d-flex flex-column bg-white">
                        {item?.text && (
                          <div className="mb-3" style={{ overflowY: item.text.length > 5 ? 'auto' : 'visible', maxHeight: '130px', scrollbarWidth: 'none', msOverflowStyle: 'none' }}>
                            <style>{`.description-scroll::-webkit-scrollbar { display: none; }`}</style>
                            <div className="description-scroll">{item.text.map((line, i) => <p key={i} className="small text-muted mb-1">{line}</p>)}</div>
                          </div>
                        )}

                        <div className="border rounded bg-light p-2" style={{ minHeight: '300px', overflowX: 'hidden', overflowY: 'auto', scrollbarWidth: 'none', msOverflowStyle: 'none' }}>
                          <style>{`.chart-scroll::-webkit-scrollbar { display: none; }`}</style>
                          {plotData && Object.keys(plotData).length ? (
                            <div className="chart-scroll" style={{ width: '100%', height: '300px' }}>
                              <Plot
                                data={plotData.data}
                                layout={{ ...plotData.layout, autosize: true, margin: { l: 50, r: 30, t: 30, b: 40 }, paper_bgcolor: '#fff', plot_bgcolor: '#fff', font: { size: 11, color: '#222' } }}
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

        {/* Fullscreen Modal */}
        <Modal show={!!modalData} onHide={() => setModalData(null)} size="xl" centered>
          <Modal.Header closeButton>
            <Modal.Title>Chart Fullscreen</Modal.Title>
          </Modal.Header>
          <Modal.Body style={{ height: '80vh' }}>
            {modalData && (
              <Plot
                data={modalData.data}
                layout={{ ...modalData.layout, autosize: true, margin: { l: 60, r: 40, t: 50, b: 50 }, paper_bgcolor: '#fff', plot_bgcolor: '#fff', font: { size: 12, color: '#222' } }}
                config={{ displayModeBar: true, responsive: true }}
                style={{ width: '100%', height: '100%' }}
              />
            )}
          </Modal.Body>
        </Modal>
      </motion.div>
    </Container>
  );
};

export default Technical;
