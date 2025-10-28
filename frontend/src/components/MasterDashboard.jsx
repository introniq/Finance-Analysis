/* MasterDashboard.jsx  –  beautiful, card-based, zero crashes, multi-symbol support */
import React, { useState } from "react";
import { Container, Row, Col, Card, Badge, ListGroup, Alert, Dropdown, Table, Tabs, Tab } from "react-bootstrap";
import { motion } from "framer-motion";
import FileDownloadHandler from "./FileDownloadHandler";
import Summary from "./Summary"; // Reuse for selected symbol details
import Volume from "./Volume";
import OIProfile from "./OIProfile";
import TPOProfile from "./TPOProfile";
import Trends from "./Trends";
import Technical from "./Technical";
import Wyckoff from "./Wyckoff";
import Periods from "./Periods";

const MasterDashboard = ({ analysisData, originalFile }) => {
  const [selectedSymbol, setSelectedSymbol] = useState(null);
  const [activeTab, setActiveTab] = useState('Summary');

  if (!analysisData || !analysisData.master_analysis) return null;

  const masterAnalysis = analysisData.master_analysis;
  const symbols = Object.keys(masterAnalysis);

  // Summary table data
  const summaryRows = symbols.map(sym => {
    const data = masterAnalysis[sym];
    const stats = data.summary_stats || {};
    return {
      symbol: sym,
      marketCap: `${Number(data.market_cap_cr || 0).toFixed(2)} Cr`,
      conditionsMet: stats.total_conditions_met || 0,
      qualifyingDates: stats.qualifying_dates_count || 0,
      delvTimes: stats.condition1_count || 0,
      amountThreshold: stats.condition2_count || 0,
      accumulation: stats.condition3_count || 0
    };
  });

  // Get selected symbol data or null
  const selectedData = selectedSymbol ? { ...analysisData, ...masterAnalysis[selectedSymbol] } : null;

  // Tab panes for selected symbol
  const tabPanes = {
    Summary: selectedData ? <Summary data={selectedData.summary || {}} /> : <Alert variant="info">Select a symbol to view details</Alert>,
    Volume: selectedData ? <Volume data={selectedData.volume || {}} /> : <Alert variant="info">Select a symbol to view details</Alert>,
    OIProfile: selectedData ? <OIProfile data={selectedData.oi_profile || {}} /> : <Alert variant="info">Select a symbol to view details</Alert>,
    TPOProfile: selectedData ? <TPOProfile data={selectedData.tpo_profile || {}} /> : <Alert variant="info">Select a symbol to view details</Alert>,
    Trends: selectedData ? <Trends stats={selectedData.trends?.pcr_stats} plot={selectedData.trends?.plot} /> : <Alert variant="info">Select a symbol to view details</Alert>,
    Technical: selectedData ? <Technical summary={selectedData.technical?.summary} plots={selectedData.technical?.plots} /> : <Alert variant="info">Select a symbol to view details</Alert>,
    Wyckoff: selectedData ? <Wyckoff overview={selectedData.wyckoff?.overview} recent={selectedData.wyckoff?.recent} plot={selectedData.wyckoff?.plot} /> : <Alert variant="info">Select a symbol to view details</Alert>,
    Periods: selectedData ? <Periods data={selectedData.periods?.data} /> : <Alert variant="info">Select a symbol to view details</Alert>,
  };

  const MetricCard = ({ icon, label, value, color = "primary" }) => (
    <motion.div whileHover={{ scale: 1.03 }}>
      <Card bg={color} text="white" className="shadow-sm h-100 border-0 rounded-4">
        <Card.Body className="d-flex flex-column align-items-center justify-content-center p-3">
          <div className="fs-3 mb-2">{icon}</div>
          <div className="small text-center">{label}</div>
          <div className="fw-bold fs-5 mt-1">{value}</div>
        </Card.Body>
      </Card>
    </motion.div>
  );

  return (
    <Container fluid className="p-3 bg-light">
      <motion.div initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        {/* Download button */}
        <Row className="mb-3">
          <Col className="d-flex justify-content-end">
            <FileDownloadHandler analysisData={analysisData} originalFile={originalFile} />
          </Col>
        </Row>

        {/* KPI row */}
        <Row className="g-3 mb-4">
          <Col xs={6} md={3}><MetricCard icon="📁" label="Total Symbols" value={symbols.length} color="info" /></Col>
          <Col xs={6} md={3}>
            <MetricCard 
              icon="✅" 
              label="Total Conditions" 
              value={symbols.reduce((sum, sym) => sum + (masterAnalysis[sym].summary_stats?.total_conditions_met || 0), 0)} 
              color="success" 
            />
          </Col>
          <Col xs={6} md={3}>
            <MetricCard 
              icon="📅" 
              label="Total Dates" 
              value={symbols.reduce((sum, sym) => sum + (masterAnalysis[sym].summary_stats?.qualifying_dates_count || 0), 0)} 
              color="warning" 
            />
          </Col>
          <Col xs={6} md={3}>
            <Dropdown>
              <Dropdown.Toggle variant="outline-primary" id="select-symbol" className="w-100">
                {selectedSymbol || 'Select Symbol'}
              </Dropdown.Toggle>
              <Dropdown.Menu className="w-100">
                {symbols.map(sym => (
                  <Dropdown.Item key={sym} onClick={() => setSelectedSymbol(sym)}>
                    {sym} ({masterAnalysis[sym].summary_stats?.total_conditions_met || 0} conditions)
                  </Dropdown.Item>
                ))}
              </Dropdown.Menu>
            </Dropdown>
          </Col>
        </Row>

        {/* Summary Table */}
        <Row className="g-3 mb-4">
          <Col xs={12}>
            <Card className="shadow-sm border-0 rounded-4">
              <Card.Header className="bg-primary text-white fw-bold">📊 Symbol Breakdown</Card.Header>
              <Card.Body>
                <Table responsive striped bordered hover size="sm">
                  <thead className="table-dark">
                    <tr>
                      <th>Symbol</th>
                      <th>Market Cap (Cr)</th>
                      <th>Conditions Met</th>
                      <th>Qualifying Dates</th>
                      <th>Delv Times {'>'} 3%</th>
                      <th>Amount {'>'} 1% MC</th>
                      <th>Accum {'>'} 2% MC</th>
                    </tr>
                  </thead>
                  <tbody>
                    {summaryRows.map((row, idx) => (
                      <tr key={idx} className={selectedSymbol === row.symbol ? 'table-active' : ''}>
                        <td><Badge bg="primary">{row.symbol}</Badge></td>
                        <td>{row.marketCap}</td>
                        <td><Badge bg="success">{row.conditionsMet}</Badge></td>
                        <td><Badge bg="warning">{row.qualifyingDates}</Badge></td>
                        <td className="text-warning">{row.delvTimes}</td>
                        <td className="text-danger">{row.amountThreshold}</td>
                        <td className="text-success">{row.accumulation}</td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              </Card.Body>
            </Card>
          </Col>
        </Row>

        {/* Selected Symbol Details (if selected) */}
        {selectedSymbol && (
          <>
            <Row className="mb-3">
              <Col>
                <h5 className="fw-bold text-primary">
                  <i className="fas fa-eye me-2" /> Details for {selectedSymbol}
                </h5>
              </Col>
            </Row>

            {/* Tabs for selected symbol */}
            <Row className="mb-3">
              <Col>
                <Tabs activeKey={activeTab} onSelect={setActiveTab} justify className="mb-3">
                  {Object.keys(tabPanes).map(k => (
                    <Tab eventKey={k} title={k} key={k}>
                      {tabPanes[k]}
                    </Tab>
                  ))}
                </Tabs>
              </Col>
            </Row>
          </>
        )}

        {/* No selection message */}
        {!selectedSymbol && (
          <Row>
            <Col className="text-center py-5">
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <i className="fas fa-search fs-1 text-muted mb-3" />
                <h5 className="text-muted">Select a symbol from the table above to view detailed analysis</h5>
              </motion.div>
            </Col>
          </Row>
        )}
      </motion.div>
    </Container>
  );
};

export default MasterDashboard;