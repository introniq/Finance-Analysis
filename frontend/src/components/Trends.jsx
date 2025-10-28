// Trends.jsx – Deep-detailed PCR × Date line chart with zones, annotations, hovers, side stats, fullscreen
import React, { useState, useMemo } from 'react';
import { Container, Button, Modal, Badge, Card, Row, Col } from 'react-bootstrap';
import Plot from 'react-plotly.js';
import { motion } from 'framer-motion';
import { FaExpand, FaChartLine, FaInfoCircle } from 'react-icons/fa';

const Trends = ({ stats, plot }) => {
  const [modalData, setModalData] = useState(null);

  // All hooks called unconditionally at top level
  const parsedPlot = useMemo(() => {
    if (!plot) return { data: [] };
    try {
      return typeof plot === 'string' ? JSON.parse(plot) : plot;
    } catch (error) {
      console.warn('Failed to parse plot:', error);
      return { data: [] };
    }
  }, [plot]);

  const figure = useMemo(() => {
    if (!stats || parsedPlot.data.length === 0 || !parsedPlot.data[0].x || !parsedPlot.data[0].y) {
      return { data: [], layout: {} };
    }

    const x = parsedPlot.data[0].x || [];
    const y = parsedPlot.data[0].y || [];
    const mean = Number(stats.mean) || 0;
    const latest = Number(stats.latest) || 0;
    const trend = stats.trend || 'Neutral';

    // Calculate std dev
    const stdDev = (arr) => {
      const n = arr.length;
      if (!n) return 0;
      const avg = arr.reduce((a, b) => a + b, 0) / n;
      return Math.sqrt(arr.reduce((sq, v) => sq + (v - avg) ** 2, 0) / n);
    };
    const std = stdDev(y.filter(v => typeof v === 'number'));

    // % change over last 5 points (deep detail)
    const recentY = y.slice(-5);
    const pctChange = recentY.length > 1 ? ((recentY[recentY.length - 1] - recentY[0]) / recentY[0] * 100) : 0;

    /* 1. Bear zone background (>1.3 PCR: high put volume, bearish) */
    const bearZone = {
      x, y: Array(x.length).fill(1.3),
      type: 'scatter',
      mode: 'lines',
      line: { color: 'rgba(0,0,0,0)' },
      fill: 'tonexty',
      fillcolor: 'rgba(231,76,60,0.12)',
      name: 'Bearish Zone (PCR > 1.3)',
      hoverinfo: 'skip',
      hovertemplate: 'Bearish: High Put Volume<extra></extra>'
    };

    /* 2. Bull zone background (<0.7 PCR: low put volume, bullish) */
    const bullZone = {
      x, y: Array(x.length).fill(0.7),
      type: 'scatter',
      mode: 'lines',
      fill: 'tonexty',
      fillcolor: 'rgba(46,204,113,0.12)',
      line: { color: 'rgba(0,0,0,0)' },
      name: 'Bullish Zone (PCR < 0.7)',
      hoverinfo: 'skip',
      hovertemplate: 'Bullish: Low Put Volume<extra></extra>'
    };

    /* 3. ±1σ band (volatility envelope) */
    const upper = mean + std;
    const lower = mean - std;
    const bandUpper = {
      x, y: Array(x.length).fill(upper),
      type: 'scatter',
      mode: 'lines',
      line: { color: 'rgba(0,0,0,0)' },
      showlegend: false,
      hoverinfo: 'skip',
    };
    const bandLower = {
      x, y: Array(x.length).fill(lower),
      type: 'scatter',
      mode: 'lines',
      fill: 'tonexty',
      fillcolor: 'rgba(149,165,166,0.15)',
      line: { color: 'rgba(0,0,0,0)' },
      name: '±1σ Band',
      hovertemplate: 'Volatility: ±%{y:.2f}<extra></extra>',
    };

    /* 4. Mean line */
    const meanLine = {
      x, y: Array(x.length).fill(mean),
      type: 'scatter',
      mode: 'lines',
      line: { color: '#95a5a6', width: 2, dash: 'dot' },
      name: `Mean: ${mean.toFixed(2)}`,
      hovertemplate: 'Historical Mean: %{y:.2f}<extra></extra>',
    };

    /* 5. Main PCR line with rich hover (% change, trend) */
    const pcrLine = {
      x, y,
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: '#667eea', width: 2.5 },
      marker: { size: 4, color: '#667eea' },
      name: 'Put-Call Ratio (PCR)',
      hovertemplate: 
        '%{x|%d-%b-%Y}<br>' +
        'PCR: %{y:.2f}<br>' +
        'Trend: %{customdata.trend}<br>' +
        '% Chg (Recent): %{customdata.pct:.1f}%<extra></extra>',
      customdata: y.map((val, idx) => ({
        trend: idx === y.length - 1 ? trend : 'N/A',
        pct: idx >= 4 ? ((y[idx] - y[idx - 4]) / y[idx - 4] * 100) : 0
      }))
    };

    const layout = {
      title: { 
        text: `PCR Trends Analysis – ${trend} Momentum (${pctChange > 0 ? 'Rising' : 'Falling'} ${Math.abs(pctChange).toFixed(1)}%)`, 
        font: { size: 16, color: '#2c3e50' } 
      },
      xaxis: { 
        title: 'Date', 
        showgrid: true, 
        gridcolor: '#ecf0f1', 
        tickformat: '%d-%b-%Y',
        hoverformat: '%d-%b-%Y'
      },
      yaxis: { 
        title: 'Put-Call Ratio (PCR)', 
        tickformat: '.2f', 
        showgrid: true, 
        gridcolor: '#ecf0f1',
        rangemode: 'tozero'
      },
      margin: { l: 60, r: 60, t: 60, b: 50 },
      hovermode: 'x unified',
      legend: { orientation: 'h', yanchor: 'bottom', y: -0.2, xanchor: 'center', x: 0.5 },
      shapes: [
        // Threshold lines
        { type: 'line', x0: x[0], x1: x[x.length - 1], y0: 0.7, y1: 0.7, line: { color: '#27ae60', width: 1, dash: 'dash' }, name: 'Bull Threshold' },
        { type: 'line', x0: x[0], x1: x[x.length - 1], y0: 1.3, y1: 1.3, line: { color: '#e74c3c', width: 1, dash: 'dash' }, name: 'Bear Threshold' }
      ],
      annotations: [
        // Latest point annotation
        {
          x: x[x.length - 1],
          y: latest,
          text: `${latest.toFixed(2)} (${trend})`,
          showarrow: true,
          arrowhead: 2,
          arrowsize: 1.2,
          arrowwidth: 2,
          arrowcolor: trend === 'Rising' ? '#27ae60' : '#e74c3c',
          ax: 20,
          ay: -30,
          font: { size: 11, color: '#2c3e50' }
        },
        // Deep detail: PCR formula & interpretation
        {
          xref: 'paper', yref: 'paper',
          x: 0.02, y: 0.98,
          text: `PCR = Put OI / Call OI | <0.7: Bullish (Call dominance) | >1.3: Bearish (Put protection) | Hover for % changes & trends`,
          showarrow: false,
          font: { size: 10, color: '#7f8c8d' },
          bgcolor: 'rgba(255,255,255,0.9)',
          bordercolor: '#bdc3c7',
          borderwidth: 1
        }
      ]
    };

    return {
      data: [bearZone, bullZone, bandLower, bandUpper, meanLine, pcrLine],
      layout,
    };
  }, [parsedPlot, stats]);

  // Early exit with loading/empty state (after all hooks)
  if (!stats || parsedPlot.data.length === 0) {
    return (
      <Container fluid className="d-flex align-items-center justify-content-center vh-100 bg-light">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center text-muted"
        >
          <div className="spinner-border text-info mb-3" style={{ width: '3rem', height: '3rem' }} role="status" />
          <h5><FaChartLine className="me-2 text-info" />No PCR Trends Data</h5>
          <p className="mb-0">Upload data with 'PCR' column to analyze Put-Call Ratio trends.</p>
        </motion.div>
      </Container>
    );
  }

  // Fullscreen Modal
  const FullscreenModal = () => (
    <Modal show={!!modalData} onHide={() => setModalData(null)} size="xl" centered>
      <Modal.Header closeButton className="bg-light">
        <Modal.Title>PCR Trends – Full Analysis</Modal.Title>
      </Modal.Header>
      <Modal.Body style={{ height: '70vh' }}>
        {modalData && (
          <Plot
            data={modalData.data}
            layout={{ ...modalData.layout, autosize: true }}
            config={{ 
              displayModeBar: true, 
              responsive: true,
              modeBarButtonsToAdd: ['drawline', 'drawrect', 'eraseshape'],
              toImageButtonOptions: { format: 'png', filename: 'pcr-trends-detailed' }
            }}
            style={{ width: '100%', height: '100%' }}
          />
        )}
      </Modal.Body>
    </Modal>
  );

  // Side Details Panel: Stats, Interpretation, Recent Table
  const SideDetails = () => (
    <Card className="h-100 shadow-sm border-0" style={{ backgroundColor: '#f8f9fa' }}>
      <Card.Header className="bg-info text-white py-2">
        <FaInfoCircle className="me-1" /> PCR Insights
      </Card.Header>
      <Card.Body className="p-3 small">
        {/* KPIs */}
        <Row className="mb-3">
          <Col xs={6}><strong>Latest PCR:</strong> {stats.latest?.toFixed(2) || 'N/A'}</Col>
          <Col xs={6}><strong>Mean:</strong> {stats.mean?.toFixed(2) || 'N/A'}</Col>
          <Col xs={6}><strong>Trend:</strong> <Badge bg={stats.trend === 'Rising' ? 'success' : 'danger'}>{stats.trend}</Badge></Col>
          <Col xs={6}><strong>Std Dev:</strong> {figure.std?.toFixed(2) || 'N/A'}</Col>
        </Row>

        {/* Interpretation */}
        <div className="mb-3">
          <strong>Interpretation:</strong>
          <p className="mb-1 text-muted">PCR measures market sentiment via open interest. Low PCR indicates call buying (bullish); high suggests put protection (bearish).</p>
          {Number(stats.latest) < 0.7 && <Badge bg="success">Bullish Signal</Badge>}
          {Number(stats.latest) > 1.3 && <Badge bg="danger">Bearish Signal</Badge>}
          {Number(stats.latest) >= 0.7 && Number(stats.latest) <= 1.3 && <Badge bg="warning">Neutral Range</Badge>}
        </div>

        {/* Recent Values Table (deep detail: last 5 points) */}
        <div className="mb-2">
          <strong>Recent PCR Values:</strong>
          <table className="table table-sm table-hover mt-1">
            <thead><tr><th>Date</th><th>PCR</th><th>% Chg</th></tr></thead>
            <tbody>
              {parsedPlot.data[0]?.x?.slice(-5).map((date, idx) => {
                const val = parsedPlot.data[0]?.y?.slice(-5)[idx];
                const prev = parsedPlot.data[0]?.y?.slice(-6)[idx];
                const chg = prev ? ((val - prev) / prev * 100).toFixed(1) : 0;
                return (
                  <tr key={idx}>
                    <td>{new Date(date).toLocaleDateString()}</td>
                    <td>{val?.toFixed(2)}</td>
                    <td className={chg > 0 ? 'text-success' : 'text-danger'}>{chg}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Formula Hint */}
        <div className="text-muted fst-italic small">
          Formula: PCR = Cumulative Put OI / Cumulative Call OI (daily). Data from last {parsedPlot.data[0]?.x?.length || 0} periods.
        </div>
      </Card.Body>
    </Card>
  );

  return (
    <Container fluid className="p-3 bg-light vh-100 d-flex flex-column">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="flex-grow-1 d-flex"
      >
        {/* Main Chart */}
        <div className="flex-grow-1 position-relative me-3" style={{ height: '100%' }}>
          <Plot
            data={figure.data}
            layout={figure.layout}
            config={{ 
              responsive: true, 
              displayModeBar: true, 
              displaylogo: false,
              staticPlot: false
            }}
            style={{ width: '100%', height: '100%' }}
            useResizeHandler
          />
          <Button
            variant="outline-info"
            size="sm"
            className="position-absolute top-0 end-0 m-3"
            onClick={() => setModalData({ title: 'PCR Trends', data: figure.data, layout: figure.layout })}
          >
            <FaExpand className="me-1" /> Fullscreen
          </Button>
        </div>

        {/* Side Details */}
        <div className="d-none d-md-block" style={{ width: '300px' }}>
          <SideDetails />
        </div>
      </motion.div>

      {/* Mobile Side Details (collapsible if needed) */}
      <div className="d-md-none mt-3">
        <SideDetails />
      </div>

      <FullscreenModal />
    </Container>
  );
};

export default Trends;