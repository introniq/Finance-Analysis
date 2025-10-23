// Navbar.jsx
import React from 'react';
import { motion } from 'framer-motion';

const Navbar = () => {
  return (
    <motion.div
      initial={{ y: -50, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="d-flex align-items-center justify-content-center"
      style={{
        color: 'white',
        width: '100%',
        height: '10vh',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        fontSize: 'clamp(1.5rem, 3vw, 2.5rem)',
        fontWeight: 'bolder',
        textAlign: 'center',
        boxShadow: '0 4px 20px rgba(102, 126, 234, 0.3)',
        position: 'relative',
        overflow: 'hidden'
      }}
    >
      <div style={{ zIndex: 1 }}>🚀 Algo Trading - OI & Delivery Analysis</div>
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent)',
        animation: 'shine 2s infinite'
      }}></div>
      <style jsx>{`
        @keyframes shine {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </motion.div>
  );
};

export default Navbar;