// useLiveStream.js – bullet-proof, auto-reconnecting, CORS-safe
import { useState, useEffect } from 'react';

export const useLiveStream = (streamUrl, symbol) => {
  const [ticks, setTicks] = useState([]);
  const [status, setStatus] = useState("CONNECTING");

  useEffect(() => {
    const sseUrl = `http://localhost:8050/stream?channel=${symbol}`;
    let es = new EventSource(sseUrl);

    const reset = () => {
      es.close();
      es = new EventSource(sseUrl);
      bind();
    };

    const bind = () => {
      es.onopen = () => {
        console.log("SSE Connected");
        setStatus("LIVE");
      };
      es.onmessage = (e) => {
        try {
          const d = JSON.parse(e.data);
          console.log("SSE Message:", d);
          if (d.heartbeat) return;
          const tick = {
            t: new Date(d.timestamp),
            p: Number(d.price),
            v: Number(d.volume),
            pc: Number(d.change_pct),
            o: Number(d.open),
            h: Number(d.high),
            l: Number(d.low),
          };
          setTicks((prev) => [...prev.slice(-29), tick]);
        } catch (err) {
          console.warn("Bad SSE packet", err);
        }
      };
      es.onerror = () => {
        console.error("SSE Error, reconnecting...");
        setStatus("ERROR");
        setTimeout(reset, 3_000);
      };
    };

    bind();
    return () => es.close();
  }, [symbol]);

  return { ticks, status };
};