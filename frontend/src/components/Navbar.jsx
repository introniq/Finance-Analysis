import React from 'react';
import { FaHome } from 'react-icons/fa';

const Navbar = () => {
  const goHome = () => {
    window.location.reload();
  };

  return (
    <div
      style={{
        color: 'rgba(255, 255, 255, 0.87)',
        width: '100%',
        height: '10vh',
        backgroundColor: 'rgba(72, 89, 132, 0.9)',
        fontSize: 'clamp(1rem, 2vw, 2rem)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontWeight: 'bolder',
        textAlign: 'center',
        position: 'relative',
        padding: '0 1rem',
      }}
    >
      <FaHome
        onClick={goHome}
        style={{
          position: 'absolute',
          left: '1rem',
          cursor: 'pointer',
          fontSize: 'clamp(1.5rem, 4vw, 3rem)',
          opacity: 0.8,
          transition: 'opacity 0.2s, transform 0.2s',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.opacity = 1;
          e.currentTarget.style.transform = 'scale(1.1)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.opacity = 0.8;
          e.currentTarget.style.transform = 'scale(1)';
        }}
      />
      Algo Trading - OI & Delivery Analysis
    </div>
  );
};

export default Navbar;
