// Periods.jsx
import React from "react";
import { Table, Card, Container, Badge, Pagination } from "react-bootstrap";
import { motion } from 'framer-motion';

const Periods = ({ data }) => {
  const [currentPage, setCurrentPage] = React.useState(1);
  const itemsPerPage = 10;

  if (!data || data.length === 0) {
    return (
      <Container fluid className="p-4">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center text-muted p-5"
        >
          <i className="fas fa-calendar-times fa-3x mb-3"></i>
          <h5>No Qualifying Period Data Available</h5>
          <p className="mb-0">Refine your analysis parameters or upload additional data to identify qualifying periods.</p>
        </motion.div>
      </Container>
    );
  }

  const tableData = data.map(row => ({
    ...row,
    ...Object.fromEntries(
      Object.entries(row).map(([k, v]) => [k.replace(/_/g, ' '), typeof v === "number" ? v.toFixed(2) : v])
    )
  }));

  const totalPages = Math.ceil(tableData.length / itemsPerPage);
  const startIdx = (currentPage - 1) * itemsPerPage;
  const currentData = tableData.slice(startIdx, startIdx + itemsPerPage);

  const handlePageChange = (pageNumber) => setCurrentPage(pageNumber);

  return (
    <Container fluid className="p-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Card className="shadow-lg border-0 rounded-4 overflow-hidden">
          <Card.Body className="p-4">
            <Card.Title className="h4 text-center mb-4" style={{ color: '#667eea', fontWeight: 'bold' }}>
              📅 Qualifying Periods Overview
              <Badge bg="primary" className="ms-2">{data.length} Periods Found</Badge>
            </Card.Title>
            <div style={{ maxHeight: "500px", overflowY: "auto", scrollbarWidth: "none", msOverflowStyle: "none" }} className="hide-scrollbar">
              <Table striped bordered hover responsive className="mb-0 rounded-3 overflow-hidden" style={{ background: 'white', fontSize: '0.9rem' }}>
                <thead className="table-dark sticky-top">
                  <tr>
                    {Object.keys(currentData[0] || {}).map((key, idx) => (
                      <th key={idx} className="fw-bold text-center py-3">{key}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {currentData.map((row, idx) => (
                    <tr key={idx} className={idx % 2 === 0 ? 'table-light' : 'table-white'}>
                      {Object.values(row).map((value, i) => (
                        <td key={i} className="text-center fw-medium py-3 align-middle">
                          {typeof value === 'number' ? value.toLocaleString() : value}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </Table>
            </div>
            <div className="d-flex justify-content-between align-items-center mt-3">
              <small className="text-muted">
                Showing {startIdx + 1} to {Math.min(startIdx + itemsPerPage, tableData.length)} of {tableData.length} periods
              </small>
              <Pagination size="sm" className="mb-0">
                <Pagination.Prev onClick={() => handlePageChange(currentPage - 1)} disabled={currentPage === 1} />
                {[...Array(totalPages)].map((_, idx) => (
                  <Pagination.Item key={idx + 1} active={idx + 1 === currentPage} onClick={() => handlePageChange(idx + 1)}>
                    {idx + 1}
                  </Pagination.Item>
                ))}
                <Pagination.Next onClick={() => handlePageChange(currentPage + 1)} disabled={currentPage === totalPages} />
              </Pagination>
            </div>
          </Card.Body>
        </Card>
      </motion.div>
      <style jsx>{`
        .hide-scrollbar::-webkit-scrollbar { 
          display: none; 
        }
        .table th, .table td {
          vertical-align: middle;
        }
      `}</style>
    </Container>
  );
};

export default Periods;