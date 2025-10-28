// Technical.jsx – Deep-detailed multi-trace line graphs with annotations, hover details, fullscreen
import React, { useState, useMemo } from 'react';
import { Container, Button, Modal, Badge, Card } from 'react-bootstrap';
import Plot from 'react-plotly.js';
import { motion } from 'framer-motion';
import { FaExpand, FaChartLine } from 'react-icons/fa';

const Technical = ({ summary, plots }) => {
  const [modalData, setModalData] = useState(null);
  const plotKeys = ['rsi', 'macd', 'bb', 'stoch', 'adx', 'vwap'];

  // All hooks called unconditionally at top level
  const parsedPlots = useMemo(() => {
    if (!plots) return {};
    try {
      return Object.fromEntries(
        Object.entries(plots).map(([key, plot]) => [
          key,
          typeof plot === 'string' ? JSON.parse(plot) : plot
        ])
      );
    } catch (error) {
      console.warn('Failed to parse plots:', error);
      return {};
    }
  }, [plots]);

  // Early exit with loading/empty state (after all hooks)
  if (!summary || Object.keys(parsedPlots).length === 0) {
    return (
      <Container fluid className="d-flex align-items-center justify-content-center vh-100 bg-light">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center text-muted"
        >
          <div className="spinner-border text-primary mb-3" style={{ width: '3rem', height: '3rem' }} role="status" />
          <h5><FaChartLine className="me-2" />No Technical Data Available</h5>
          <p className="mb-0">Upload a valid file and run analysis to view RSI, MACD, and more.</p>
        </motion.div>
      </Container>
    );
  }

  // Enhanced Chart Card: Renders full Plotly figure + deep details
  const ChartCard = ({ title, item, plotData, key }) => {
    // All hooks called unconditionally at top level inside ChartCard
    const enhancedLayout = useMemo(() => {
      if (!plotData?.data?.length || !plotData.layout) {
        return {
          title: { text: title, font: { size: 14, color: '#2c3e50' } },
          xaxis: { title: 'Date', showgrid: true, gridcolor: '#ecf0f1' },
          yaxis: { title: 'Value', showgrid: true, gridcolor: '#ecf0f1' },
          margin: { l: 60, r: 40, t: 50, b: 50 },
          hovermode: 'x unified',
          showlegend: true,
          shapes: [],
          annotations: []
        };
      }

      // Enhance layout with deep details (shapes, annotations, hover)
      return {
        ...plotData.layout,
        title: {
          text: `${title} Analysis ${item?.badge ? `– ${item.badge}` : ''}`,
          font: { size: 14, color: '#2c3e50' }
        },
        xaxis: {
          ...plotData.layout?.xaxis,
          title: 'Date',
          showgrid: true,
          gridcolor: '#ecf0f1',
          tickformat: '%d-%b-%Y',
          hoverformat: '%d-%b-%Y'
        },
        yaxis: {
          ...plotData.layout?.yaxis,
          title: item?.text?.[0]?.split(':')[0] || 'Value',
          showgrid: true,
          gridcolor: '#ecf0f1',
          tickformat: '.2f'
        },
        margin: { l: 60, r: 40, t: 50, b: 50 },
        hovermode: 'x unified',
        showlegend: true,
        shapes: [
          // Example: Add signal lines (customize per indicator)
          ...(title.includes('RSI') ? [
            { type: 'line', x0: plotData.layout?.xaxis?.range?.[0] || 0, x1: 'last', y0: 70, y1: 70, line: { color: 'red', width: 1, dash: 'dash' }, name: 'Overbought' },
            { type: 'line', x0: plotData.layout?.xaxis?.range?.[0] || 0, x1: 'last', y0: 30, y1: 30, line: { color: 'green', width: 1, dash: 'dash' }, name: 'Oversold' },
            { type: 'line', x0: plotData.layout?.xaxis?.range?.[0] || 0, x1: 'last', y0: 50, y1: 50, line: { color: 'gray', width: 1, dash: 'dot' }, name: 'Neutral' }
          ] : []),
          ...(title.includes('MACD') ? [
            { type: 'line', x0: plotData.layout?.xaxis?.range?.[0] || 0, x1: 'last', y0: 0, y1: 0, line: { color: 'gray', width: 1, dash: 'dot' }, name: 'Zero Line' }
          ] : [])
        ],
        annotations: [
          // Latest value annotation
          {
            x: plotData.data[0].x?.slice(-1)[0],
            y: plotData.data[0].y?.slice(-1)[0],
            text: `${plotData.data[0].y?.slice(-1)[0]?.toFixed(2)} ${item?.badge || ''}`,
            showarrow: true,
            arrowhead: 2,
            arrowsize: 1,
            arrowwidth: 2,
            arrowcolor: '#3498db',
            ax: 0,
            ay: -30,
            font: { size: 10 }
          },
          // Deep detail tooltip hint
          {
            xref: 'paper', yref: 'paper',
            x: 0.02, y: 0.98,
            text: item?.text?.[1] || `Formula: ${title} (14-period) – Hover for details`,
            showarrow: false,
            font: { size: 9, color: '#7f8c8d' },
            bgcolor: 'rgba(255,255,255,0.8)'
          }
        ]
      };
    }, [plotData, title, item]);

    const enhancedData = useMemo(() => {
      if (!plotData?.data?.length) {
        return [];
      }

      // Enhance traces with rich hover templates
      return plotData.data.map(trace => ({
        ...trace,
        hovertemplate: trace.hovertemplate || 
          `${trace.name || title}: %{y:.4f}<br>Date: %{x|%d-%b-%Y}<extra></extra>`,
        line: { ...trace.line, width: 2.5 }
      }));
    }, [plotData, title]);

    // Early return after hooks (now safe)
    if (!plotData?.data?.length) return null;

    return (
      <Card className="h-100 shadow-sm border-0" style={{ backgroundColor: '#f8f9fa' }}>
        <Card.Header className="d-flex justify-content-between align-items-center py-2 px-3 bg-gradient-primary text-white small">
          <span><FaChartLine className="me-1" />{title}</span>
          {item?.badge && <Badge bg={item.color || 'info'}>{item.badge}</Badge>}
        </Card.Header>
        <Card.Body className="p-2 d-flex flex-column">
          {item?.text && (
            <div className="mb-2 small text-muted">
              {item.text.map((t, i) => <div key={i} className="mb-1">{t}</div>)}
            </div>
          )}
          <div className="flex-grow-1 position-relative" style={{ height: '280px' }}>
            <Plot
              data={enhancedData}
              layout={enhancedLayout}
              config={{ 
                displayModeBar: false, 
                responsive: true, 
                staticPlot: false,
                toImageButtonOptions: { format: 'png', filename: `${title.toLowerCase()}-chart` }
              }}
              style={{ width: '100%', height: '100%' }}
              useResizeHandler
              onHover={(data) => console.log('Hover:', data.points[0])} // Debug hook
            />
            <Button
              variant="outline-primary"
              size="sm"
              className="position-absolute top-0 end-0 m-2"
              onClick={() => setModalData({ title, data: enhancedData, layout: enhancedLayout })}
            >
              <FaExpand />
            </Button>
          </div>
        </Card.Body>
      </Card>
    );
  };

  // Fullscreen Modal with enhanced controls
  const FullscreenModal = () => (
    <Modal show={!!modalData} onHide={() => setModalData(null)} size="xl" centered dialogClassName="fullscreen-modal">
      <Modal.Header closeButton className="bg-light">
        <Modal.Title>{modalData?.title} – Full View</Modal.Title>
      </Modal.Header>
      <Modal.Body className="p-0" style={{ height: '70vh' }}>
        {modalData && (
          <Plot
            data={modalData.data}
            layout={{ ...modalData.layout, autosize: true, height: '100%' }}
            config={{ 
              displayModeBar: true, 
              responsive: true, 
              modeBarButtonsToAdd: ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
              toImageButtonOptions: { format: 'svg', filename: `${modalData.title.toLowerCase()}-detailed` }
            }}
            style={{ width: '100%', height: '100%' }}
          />
        )}
      </Modal.Body>
    </Modal>
  );

  // Main responsive grid
  return (
    <Container fluid className="p-3 bg-light vh-100 d-flex flex-column">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, staggerChildren: 0.1 }}
        className="flex-grow-1 overflow-auto"
        style={{ scrollbarWidth: 'thin', scrollbarColor: '#bdc3c7 #f8f9fa' }}
      >
        <style>{`
          .hide-scroll::-webkit-scrollbar { width: 6px; }
          .hide-scroll::-webkit-scrollbar-track { background: #f8f9fa; }
          .hide-scroll::-webkit-scrollbar-thumb { background: #bdc3c7; border-radius: 3px; }
        `}</style>
        <div className="row g-3 h-100">
          {plotKeys.map((key, idx) => {
            const item = summary[idx];
            const plotData = parsedPlots[key];
            return (
              <motion.div
                key={key}
                className="col-md-6 col-lg-4"
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: idx * 0.1 }}
              >
                <ChartCard title={item?.title || key.toUpperCase()} item={item} plotData={plotData} key={key} />
              </motion.div>
            );
          })}
        </div>
      </motion.div>
      <FullscreenModal />
    </Container>
  );
};

export default Technical;