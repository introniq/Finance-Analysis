// Wyckoff.jsx – Deep-detailed price × date line graph with event markers, annotations, hover formulas
import React, { useState, useMemo } from 'react';
import { Container, Button, Modal, Badge } from 'react-bootstrap';
import Plot from 'react-plotly.js';
import { motion } from 'framer-motion';
import { FaExpand, FaChartLine, FaBullseye } from 'react-icons/fa';

const Wyckoff = ({ overview, recent, plot }) => {
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

  const wyckoffEvents = useMemo(() => {
    // Assume events are in plot.data or overview; fallback to sample if missing
    const eventsData = overview?.events || []; // Backend should provide this; simulate if not
    if (eventsData.length === 0 || !parsedPlot.data[0]) {
      // Fallback: Parse from plot data annotations or generate samples
      const x = parsedPlot.data[0]?.x || [];
      const y = parsedPlot.data[0]?.y || [];
      return x.map((date, idx) => ({
        date,
        price: y[idx] || 0,
        type: recent || 'Neutral', // Use recent event
        phase: 'Accumulation/Distribution'
      })).filter((_, idx) => idx % 5 === 0); // Sample every 5th point
    }
    return eventsData;
  }, [parsedPlot, overview, recent]);

  const traces = useMemo(() => {
    const x = parsedPlot.data[0]?.x || [];
    const y = parsedPlot.data[0]?.y || [];

    // 1. Main price line (deep hover with % change)
    const priceLine = {
      x, y,
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: '#2c3e50', width: 3 },
      marker: { size: 4, color: '#34495e' },
      name: 'Close Price',
      hovertemplate: 
        'Wyckoff Price: ₹%{y:.2f}<br>Date: %{x|%d-%b-%Y}<br>' +
        '% Change (5d): %{customdata:.1f}%<extra></extra>',
      customdata: y.map((val, idx) => 
        idx >= 5 ? ((y[idx] - y[idx - 5]) / y[idx - 5] * 100) : 0 // 5-day % change
      )
    };

    // 2. Event markers (diamonds with colors and deep hover)
    const eventTraces = wyckoffEvents.map((ev, evIdx) => ({
      type: 'scatter',
      mode: 'markers+text',
      x: [ev.date],
      y: [ev.price],
      marker: {
        color: ev.type.includes('Strength') || ev.type === 'Spring' ? '#27ae60' : 
               ev.type.includes('Weakness') || ev.type === 'Upthrust' ? '#e74c3c' : '#f39c12',
        size: 14,
        symbol: 'diamond',
        line: { width: 2, color: 'white' }
      },
      text: [ev.type],
      textposition: 'top center',
      textfont: { size: 10, color: '#2c3e50', family: 'Arial Black' },
      name: ev.type,
      hovertemplate: 
        `Event: %{text}<br>` +
        `Price: ₹%{y:.2f}<br>` +
        `Phase: %{customdata}<br>` + // e.g., "Accumulation"
        `Implication: %{marker.color === '#27ae60' ? 'Bullish Reversal' : 'Bearish Warning'}<extra></extra>`,
      customdata: [ev.phase || 'Accumulation/Distribution'] // Deep detail: Phase
    }));

    return [priceLine, ...eventTraces];
  }, [parsedPlot, wyckoffEvents]);

  const layout = useMemo(() => {
    if (traces.length === 0 || !overview) {
      return {
        title: { text: 'Wyckoff Method – Price Action & Event Timeline', font: { size: 16, color: '#2c3e50' } },
        xaxis: { title: 'Date', showgrid: true, gridcolor: '#ecf0f1' },
        yaxis: { title: 'Price (₹)', tickformat: '.2f', showgrid: true, gridcolor: '#ecf0f1' },
        margin: { l: 60, r: 60, t: 60, b: 50 },
        hovermode: 'x unified',
        showlegend: true,
        shapes: [],
        annotations: []
      };
    }

    // Enhanced layout with shapes, annotations, and deep details
    return {
      title: { 
        text: 'Wyckoff Method – Price Action & Event Timeline', 
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
        title: 'Price (₹)', 
        tickformat: '.2f', 
        showgrid: true, 
        gridcolor: '#ecf0f1'
      },
      margin: { l: 60, r: 60, t: 60, b: 50 },
      hovermode: 'x unified',
      showlegend: true,
      shapes: [
        // Trend line example (connect first/last if bullish)
        ...(recent?.includes('Strength') ? [{
          type: 'line',
          x0: traces[0]?.x[0], x1: traces[0]?.x?.slice(-1)[0],
          y0: traces[0]?.y[0], y1: traces[0]?.y?.slice(-1)[0],
          line: { color: '#27ae60', width: 2, dash: 'dash' },
          name: 'Bullish Trend'
        }] : [])
      ],
      annotations: [
        // Recent event annotation
        {
          x: traces[0]?.x?.slice(-1)[0],
          y: traces[0]?.y?.slice(-1)[0],
          text: `${recent} (Latest)`,
          showarrow: true,
          arrowhead: 3,
          arrowsize: 1.5,
          arrowwidth: 3,
          arrowcolor: '#e67e22',
          ax: 20, ay: 20,
          font: { size: 12, color: '#e67e22' }
        },
        // Deep detail: Wyckoff overview formula/hint
        {
          xref: 'paper', yref: 'paper',
          x: 0.02, y: 0.98,
          text: `${overview} | Formula: Volume > 20d Avg & Price vs MA5 (Hover events for phases)`,
          showarrow: false,
          font: { size: 10, color: '#7f8c8d' },
          bgcolor: 'rgba(255,255,255,0.9)',
          bordercolor: '#bdc3c7',
          borderwidth: 1
        }
      ]
    };
  }, [traces, overview, recent]);

  // Early exit with loading/empty state (after all hooks)
  if (!overview || parsedPlot.data.length === 0) {
    return (
      <Container fluid className="d-flex align-items-center justify-content-center vh-100 bg-light">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center text-muted"
        >
          <div className="spinner-border text-success mb-3" style={{ width: '3rem', height: '3rem' }} role="status" />
          <h5><FaBullseye className="me-2 text-success" />No Wyckoff Data</h5>
          <p className="mb-0">Analysis detects accumulation/distribution phases – upload data to view.</p>
        </motion.div>
      </Container>
    );
  }

  // Fullscreen Modal
  const FullscreenModal = () => (
    <Modal show={!!modalData} onHide={() => setModalData(null)} size="xl" centered>
      <Modal.Header closeButton className="bg-light">
        <Modal.Title>Wyckoff Full Analysis</Modal.Title>
      </Modal.Header>
      <Modal.Body style={{ height: '70vh' }}>
        {modalData && (
          <Plot
            data={modalData.data}
            layout={{ ...modalData.layout, autosize: true }}
            config={{ 
              displayModeBar: true, 
              responsive: true,
              modeBarButtonsToAdd: ['drawline'],
              toImageButtonOptions: { format: 'png', filename: 'wyckoff-detailed' }
            }}
            style={{ width: '100%', height: '100%' }}
          />
        )}
      </Modal.Body>
    </Modal>
  );

  return (
    <Container fluid className="p-3 bg-light vh-100 d-flex flex-column">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="flex-grow-1"
      >
        <div className="position-relative" style={{ height: '100%' }}>
          <Plot
            data={traces}
            layout={layout}
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
            variant="outline-success"
            size="sm"
            className="position-absolute top-0 end-0 m-3"
            onClick={() => setModalData({ title: 'Wyckoff Timeline', data: traces, layout })}
          >
            <FaExpand className="me-1" /> Fullscreen
          </Button>
        </div>
      </motion.div>

      {/* Summary Strip with Badges */}
      <div className="d-flex justify-content-center gap-3 mt-3 p-2 bg-white border-top small">
        <Badge bg="info" className="d-flex align-items-center">
          <FaChartLine className="me-1" /> Recent: {recent || 'Neutral'}
        </Badge>
        <Badge bg="secondary" className="d-flex align-items-center">
          <FaBullseye className="me-1" /> Events: {wyckoffEvents.length}
        </Badge>
        <Badge bg="light" className="text-dark">{overview}</Badge>
      </div>

      <FullscreenModal />
    </Container>
  );
};

export default Wyckoff;