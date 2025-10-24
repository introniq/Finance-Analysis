import React, { useState, useEffect } from "react";
import FileUploader from "./components/Fileuploader";
import History from "./components/History";
import Navbar from "./components/Navbar";
import { Container, Row, Col, Button } from "react-bootstrap";

function App() {
  const [showHistory, setShowHistory] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [historyFile, setHistoryFile] = useState(null);

  useEffect(() => {
    const checkOrientation = () => {
      setIsMobile(window.innerWidth < window.innerHeight);
    };
    checkOrientation();
    window.addEventListener("resize", checkOrientation);
    return () => window.removeEventListener("resize", checkOrientation);
  }, []);

  const handleRunAnalysisFromHistory = (filename) => {
    setHistoryFile(filename);
  };

  return (
    <div className="App">
      {/* Fixed Navbar */}
      <div
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          zIndex: 999,
          backgroundColor: "rgb(128,128,128)",
        }}
      >
        <Navbar />
      </div>

      {/* Main Content */}
      <Container
        fluid
        style={{
          marginTop: "10vh",
          height: "auto",
          overflowY: "hidden",
        }}
      >
        <Row>
          {/* History Section */}
          {!isMobile ? (
            <Col xs={12} md={8} lg={3} style={{ height: "90vh" }}>
              <History onRunAnalysis={handleRunAnalysisFromHistory} /> {/* ✅ FIXED */}
            </Col>
          ) : (
            <>
              <Button
                onClick={() => setShowHistory(!showHistory)}
                style={{
                  position: "fixed",
                  bottom: "2vh",
                  left: "2vw",
                  height: "10vh",
                  width: "10vh",
                  borderRadius: "50%",
                  zIndex: 1000,
                  backgroundColor: "rgba(72, 89, 132, 0.9)",
                  border: "none",
                  fontWeight: "bold",
                }}
              >
                {showHistory ? "Close" : "Show"}
              </Button>

              {showHistory && (
                <div
                  style={{
                    position: "fixed",
                    top: "10vh",
                    left: 0,
                    right: 0,
                    bottom: 0,
                    backgroundColor: "rgba(0,0,0,0.8)",
                    zIndex: 999,
                    overflowY: "auto",
                  }}
                >
                  <History onRunAnalysis={handleRunAnalysisFromHistory} />
                </div>
              )}
            </>
          )}

          {/* File Uploader */}
          <Col
            xs={12}
            md={8}
            lg={9}
            style={{
              height: "auto",
              overflowY: "scroll",
              scrollbarWidth: "none",
              msOverflowStyle: "none",
              paddingLeft: "1rem",
              marginBottom: "1vh",
            }}
          >
            <FileUploader historyFile={historyFile} />
          </Col>
        </Row>
      </Container>
    </div>
  );
}

export default App;
