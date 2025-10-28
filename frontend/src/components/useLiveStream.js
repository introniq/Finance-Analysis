import { useState, useEffect, useCallback } from 'react';

export const useLiveStream = (symbol = "DABUR.NS") => {
  const [ticks, setTicks] = useState([]);
  const [status, setStatus] = useState("CONNECTING");

  // OPTIMIZED: Debounced tick update (limit to last 30 for performance)
  const addTick = useCallback((newTick) => {
    setTicks(prev => {
      const updated = [...prev.slice(-29), newTick]; // Keep last 30
      return updated;
    });
  }, []);

  useEffect(() => {
    let reconnectAttempts = 0;
    const maxReconnects = 5;
    const sseUrl = `http://localhost:8050/stream?channel=${symbol}`;
    let es;

    const reset = (delay = 1000) => {
      if (reconnectAttempts >= maxReconnects) {
        setStatus("ERROR");
        return;
      }
      reconnectAttempts++;
      setTimeout(() => {
        if (es) es.close();
        es = new EventSource(sseUrl);
        bind(es);
      }, delay * reconnectAttempts); // Exponential backoff
    };

    const bind = (eventSource) => {
      eventSource.onopen = () => {
        console.log("SSE Connected for", symbol);
        setStatus("LIVE");
        reconnectAttempts = 0;
      };
      eventSource.onmessage = (e) => {
        // REDUCED LOGGING: Only log errors/warnings
        try {
          const d = JSON.parse(e.data);
          if (d.error) {
            console.error("Server error in stream:", d.error);
            setStatus("ERROR");
            return;
          }
          if (d.heartbeat) {
            // console.log("Heartbeat received"); // Reduced
            return;
          }
          if (!d.price || isNaN(d.price) || d.price <= 0) {
            console.warn("Invalid price in payload:", d);
            return;
          }
          const tick = {
            t: new Date(d.timestamp || Date.now()),
            p: Number(d.price),
            v: Number(d.volume) || 0,
            pc: Number(d.change_pct) || 0,
            o: Number(d.open) || d.price,
            h: Number(d.high) || d.price,
            l: Number(d.low) || d.price,
          };
          addTick(tick);
        } catch (err) {
          console.warn("Bad SSE packet for", symbol, err);
        }
      };
      eventSource.onerror = (err) => {
        console.error("SSE Error for", symbol, ", reconnecting...");
        setStatus("CONNECTING");
        reset();
      };
    };

    try {
      es = new EventSource(sseUrl);
      bind(es);
    } catch (err) {
      console.error("Failed to create EventSource for", symbol, ":", err);
      setStatus("ERROR");
      reset(2000);
    }

    return () => {
      if (es) es.close();
    };
  }, [symbol, addTick]);

  return { ticks, status };
};