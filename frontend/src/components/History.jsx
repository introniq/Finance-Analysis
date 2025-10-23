// History.jsx
import React, { useState, useEffect } from 'react';
import { Card, OverlayTrigger, Tooltip, Container } from "react-bootstrap";
import { motion } from 'framer-motion';

const History = () => {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const saved = localStorage.getItem('analysisHistory');
    if (saved) setHistory(JSON.parse(saved));
  }, []);

  const addToHistory = (fileName, date) => {
    const newItem = { id: Date.now(), fileName, date };
    const updated = [newItem, ...history.slice(0, 9)];
    setHistory(updated);
    localStorage.setItem('analysisHistory', JSON.stringify(updated));
  };

  // Expose addToHistory to window for external use
  useEffect(() => {
    window.addToHistory = addToHistory;
  }, []);

  return (
    <Container fluid className="h-100 p-3" style={{ background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)' }}>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="rounded-3 shadow-lg"
        style={{
          height: '88vh',
          display: 'flex',
          flexDirection: 'column',
          overflowY: "scroll",
          scrollbarWidth: "none",
          msOverflowStyle: "none",
          background: 'white'
        }}
      >
        <h5 className="mb-4 fw-bold text-dark p-3 border-bottom" style={{ flex: '0 0 auto', color: '#667eea' }}>
          📁 File Analysis History
        </h5>
        <div style={{ flex: '1 1 auto', overflowY: "scroll", scrollbarWidth: "none", msOverflowStyle: "none" }} className="p-3">
          {history.length === 0 ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center text-muted mt-5"
            >
              <i className="fas fa-history fa-3x mb-3"></i>
              <h6>No analysis history yet.</h6>
              <p>Start an analysis to see your recent files here.</p>
            </motion.div>
          ) : (
            history.map((item, index) => (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <OverlayTrigger
                  placement="top"
                  overlay={<Tooltip>{item.date}</Tooltip>}
                >
                  <Card
                    className="mb-3 history-card rounded-3 border-0 shadow-sm cursor-pointer"
                    style={{
                      transition: "all 0.3s ease",
                      background: 'linear-gradient(145deg, #ffffff, #f8f9fa)',
                      minHeight: "70px",
                      display: "flex",
                      flexDirection: "column",
                      justifyContent: "center",
                      transform: 'translateX(0)'
                    }}
                  >
                    <Card.Body className="p-3">
                      <Card.Text className="fw-semibold text-dark mb-1 text-truncate">
                        📄 {item.fileName}
                      </Card.Text>
                      <Card.Text className="text-muted small mb-0">
                        Analyzed on: {new Date(item.date).toLocaleDateString()}
                      </Card.Text>
                    </Card.Body>
                  </Card>
                </OverlayTrigger>
              </motion.div>
            ))
          )}
        </div>
        <style jsx>{`
          .history-card:hover {
            transform: translateX(5px) !important;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15) !important;
            border-left: 4px solid #667eea !important;
          }
          .cursor-pointer {
            cursor: pointer;
          }
        `}</style>
      </motion.div>
    </Container>
  );
};

export default History;