// Analysis.jsx
import { useState } from 'react';
import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';
import Summary        from './Summary';
import Volume         from './Volume';
import OIProfile      from './OIProfile';
import TPOProfile     from './TPOProfile';
import Clustering     from './Clustering';
import Trends         from './Trends';
import Technical      from './Technical';
import Wyckoff        from './Wyckoff';
import Periods        from './Periods';
import Predictive     from './Predictive';
import { useLiveStream } from './useLiveStream';
import { Card, Container, Row, Col } from 'react-bootstrap';
import { motion } from 'framer-motion';

const Analysis = ({ results, file }) => {  // FIXED: Receive file prop from parent (Fileuploader)
  const [key, setKey] = useState('Summary');
  const { liveData } = useLiveStream(
    results?.stream_url,
    results?.summary?.metrics?.symbol
  );

  /* ----------  loading stub  ---------- */
  if (!results)
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="d-flex justify-content-center align-items-center vh-100"
      >
        <div className="spinner-border text-primary" role="status" style={{ width: '3rem', height: '3rem' }}>
          <span className="visually-hidden">Loading analysis…</span>
        </div>
      </motion.div>
    );

  /* ----------  reusable card wrapper  ---------- */
  const DashCard = ({ children, title }) => (
    <Card className="shadow-sm border-0 h-100">
      <Card.Header className="fw-bold bg-primary text-white py-3">{title}</Card.Header>
      <Card.Body className="p-4">{children}</Card.Body>
    </Card>
  );

  /* ----------  tab content wrappers (kept tiny so every tab looks identical)  ---------- */
  const panes = {
    Summary: (
      <DashCard title="Summary">
        <Summary data={results.summary} live={liveData} />
      </DashCard>
    ),
    Volume: (
      <DashCard title="Volume Profile">
        {/* FIXED: Pass file prop to Volume for re-fetching (e.g., peak-diff picker) */}
        <Volume data={results.volume} file={file} onData={(data) => console.log('Volume data updated:', data)} />
      </DashCard>
    ),
    OIProfile: (
      <DashCard title="Open-Interest Profile">
        <OIProfile data={results.oi_profile} />
      </DashCard>
    ),
    TPOProfile: (
      <DashCard title="TPO Profile">
        <TPOProfile data={results.tpo_profile} />
      </DashCard>
    ),
    Clustering: (
      <DashCard title="Clustering">
        <Clustering plot={results.clustering?.plot} />
      </DashCard>
    ),
    Trends: (
      <DashCard title="Trends">
        <Trends stats={results.trends?.pcr_stats} plot={results.trends?.plot} live={liveData} />
      </DashCard>
    ),
    Technical: (
      <DashCard title="Technical">
        <Technical summary={results.technical?.summary} plots={results.technical?.plots} />
      </DashCard>
    ),
    Wyckoff: (
      <DashCard title="Wyckoff">
        <Wyckoff overview={results.wyckoff?.overview} recent={results.wyckoff?.recent} plot={results.wyckoff?.plot} />
      </DashCard>
    ),
    Periods: (
      <DashCard title="Periods">
        <Periods data={results.periods?.data} />
      </DashCard>
    ),
    Predictive: (
      <DashCard title="Predictive">
        <Predictive />
      </DashCard>
    ),
  };

  /* ----------  render  ---------- */
  return (
    <Container fluid className="d-flex flex-column bg-light">
        <Tabs
          id="analysis-tabs"
          activeKey={key}
          onSelect={(k) => setKey(k)}
          justify
          className="mb-3 flex-nowrap overflow-none"
        >
          {Object.keys(panes).map((k) => (
            <Tab
              key={k}
              eventKey={k}
              title={
                <span className="d-flex align-items-center gap-1">
                  <i className={`fa fa-${iconMap[k]}`} />
                  <span className="d-none d-sm-inline">{k}</span>
                </span>
              }
            />
          ))}
        </Tabs>

      {/* ----- tab pane ----- */}
      <Row className="flex-grow-1 overflow-hidden">
        <Col xs={12} className="h-100">
          <motion.div
            key={key} // re-animate on tab switch
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.25 }}
            className="h-100"
          >
            {panes[key]}
          </motion.div>
        </Col>
      </Row>

      {/* ----- tiny inline styles ----- */}
      <style jsx global>{`
        .nav-pills .nav-link {
          color: #495057;
          background-color: #e9ecef;
          border-radius: 1rem;
          font-weight: 500;
          white-space: nowrap;
          padding: .45rem .9rem;
          transition: all .2s ease;
        }
        .nav-pills .nav-link.active {
          background-color: #0d6efd;
          color: #fff;
          transform: scale(1.05);
        }
        .nav-pills .nav-link:hover {
          background-color: #dee2e6;
        }
      `}</style>
    </Container>
  );
};

/* ----------  icon map  ---------- */
const iconMap = {
  Summary: 'chart-bar',
  Volume: 'volume-up',
  OIProfile: 'exchange-alt',
  TPOProfile: 'clock',
  Clustering: 'braille',
  Trends: 'trending-up',
  Technical: 'cogs',
  Wyckoff: 'target',
  Periods: 'calendar-alt',
  Predictive: 'crystal-ball',
};

export default Analysis;