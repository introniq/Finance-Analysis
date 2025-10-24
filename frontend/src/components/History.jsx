import React, { useState, useEffect } from 'react';
import { Card, OverlayTrigger, Tooltip, Container } from "react-bootstrap";
import { motion } from 'framer-motion';

const SERVER_BASE = "http://localhost:8051";

const History = ({ onRunAnalysis }) => {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await fetch(`${SERVER_BASE}/list-files`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const files = await response.json();
        setHistory(files);
      } catch (err) {
        console.error("Failed to fetch history:", err);
      }
    };
    fetchHistory();
  }, []);

  return (
    <Container
      fluid
      className="p-3"
      style={{
        marginTop: '1vh',
        backgroundColor: "#f1f3f6",
        borderRadius: "12px",
        boxShadow: "0 4px 20px rgba(0,0,0,0.05)",
        border: "1px solid #dcdcdc",
        height: "88vh",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden"
      }}
    >
      <h5 className="mb-3 fw-bold text-dark">File Analysis History</h5>

      <div
        style={{
          flex: '1 1 auto',
          overflowY: 'scroll',
          overflowX: 'hidden',
          scrollbarWidth: 'none',
          msOverflowStyle: 'none'
        }}
        className="history-scroll"
      >
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
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
            >
              <OverlayTrigger
                placement="top"
                overlay={<Tooltip>{item.filename}</Tooltip>}
              >
                <Card
                  className="mb-2 history-card"
                  onClick={() => onRunAnalysis && onRunAnalysis(item.filename)}
                  style={{
                    cursor: "pointer",
                    transition: "all 0.3s",
                    backgroundColor: "#ffffff",
                    border: "1px solid #e0e0e0",
                    borderRadius: "8px",
                    padding: "0.5rem 0.8rem",
                    minHeight: "60px",
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "center",
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis"
                  }}
                >
                  <Card.Text className="mb-1 fw-semibold text-dark text-truncate">
                    📄 {item.filename}
                  </Card.Text>
                  <Card.Text className="text-muted small mb-0">
                    Saved on: {new Date(item.timestamp * 1000).toLocaleString()}
                  </Card.Text>
                </Card>
              </OverlayTrigger>
            </motion.div>
          ))
        )}
      </div>

      <style>
        {`
          .history-scroll::-webkit-scrollbar {
            width: 0px;
            background: transparent;
          }

          .history-card:hover {
            transform: translateX(3px);
            box-shadow: 0 6px 20px rgba(0,123,255,0.25);
            border-color: #0d6efd;
            background-color: #f0f8ff;
            overflow: visible;
          }

          .history-card:active {
            transform: translateX(1px);
            box-shadow: 0 3px 12px rgba(0,123,255,0.15);
          }
        `}
      </style>
    </Container>
  );
};

export default History;
