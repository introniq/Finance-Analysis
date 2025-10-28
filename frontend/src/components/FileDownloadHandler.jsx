import React, { useState } from 'react';
import { Button, Modal, Alert, Spinner } from 'react-bootstrap';
import { saveAs } from 'file-saver';
import * as XLSX from 'xlsx';
import { parseISO, format, isValid } from 'date-fns';  // For robust date handling

const FileDownloadHandler = ({ analysisData, originalFile }) => {
  const [showModal, setShowModal] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState(null);

  // Robust date normalization (handles ISO, YYYY-MM-DD, timestamps)
  const normalizeDate = (dateValue) => {
    if (!dateValue) return null;
    let date;
    if (typeof dateValue === 'string') {
      date = parseISO(dateValue);
      if (!isValid(date)) {
        date = new Date(dateValue.replace(/T.*/, ''));  // Strip time
      }
    } else if (dateValue instanceof Date) {
      date = dateValue;
    } else {
      date = new Date(dateValue);
    }
    return isValid(date) ? format(date, 'yyyy-MM-dd') : null;
  };

  const downloadHighlightedExcel = async () => {
    if (!analysisData || !analysisData.master_analysis) {
      setError('No master analysis data available');
      return;
    }

    setDownloading(true);
    setError(null);

    try {
      const masterAnalysis = analysisData.master_analysis;
      const isMultiSymbol = typeof masterAnalysis === 'object' && !Array.isArray(masterAnalysis) && Object.keys(masterAnalysis).length > 1;

      // ARGB color map (8-hex: FF + RRGGBB for opacity/full color)
      const colorMap = {
        'amount_1pct_mc': {  // Red: Single-day >1% MC (highest priority)
          fill: { patternType: "solid", fgColor: { rgb: "FFFFCDD2" } }
        },
        'accumulation_2pct_mc': {  // Green: 15-day accum >2% MC
          fill: { patternType: "solid", fgColor: { rgb: "FFC8E6C9" } }
        },
        'delivery_times_3pct': {  // Yellow: 2-3 day delv >3%
          fill: { patternType: "solid", fgColor: { rgb: "FFFFE082" } }
        },
        'total_accumulation_1pct_mc': {  // Light Blue: Base for total >1% MC
          fill: { patternType: "solid", fgColor: { rgb: "FFE1F5FE" } }
        }
      };

      const processSymbolData = (symbol, symData) => {
        const rawData = symData.raw_data || [];
        if (rawData.length === 0) {
          console.warn(`No data for ${symbol}`);
          return null;
        }

        const headers = Object.keys(rawData[0]);
        const dateColIndex = headers.indexOf('Date');
        if (dateColIndex === -1) return null;

        const conditions = symData.conditions_met || [];
        console.log(`[DEBUG] ${symbol}: ${conditions.length} conditions to process`);

        const worksheet = XLSX.utils.json_to_sheet(rawData);
        const range = XLSX.utils.decode_range(worksheet['!ref']);

        // Base blue: Apply to ALL data rows if total accum met
        const totalAccMet = conditions.some(c => c.condition === 'total_accumulation_1pct_mc');
        if (totalAccMet) {
          console.log(`[DEBUG] ${symbol}: Base blue applied (total accum met)`);
          for (let R = range.s.r + 1; R <= range.e.r; ++R) {
            for (let C = range.s.c; C <= range.e.c; ++C) {
              const addr = XLSX.utils.encode_cell({ r: R, c: C });
              if (!worksheet[addr]) continue;
              worksheet[addr].s = { ... (worksheet[addr].s || {}), ...colorMap['total_accumulation_1pct_mc'] };
            }
          }
        }

        // Per-row: Override with highest priority
        let coloredCount = 0;
        for (let R = range.s.r + 1; R <= range.e.r; ++R) {
          const dateAddr = XLSX.utils.encode_cell({ r: R, c: dateColIndex });
          const dateCell = worksheet[dateAddr];
          if (!dateCell?.v) continue;

          const rowDate = normalizeDate(dateCell.v);
          if (!rowDate) continue;

          let applicable = [];

          // Red: Single-day match
          conditions.filter(c => c.condition === 'amount_1pct_mc' && rowDate === normalizeDate(c.date)).forEach(() => applicable.push('amount_1pct_mc'));

          // Green/Yellow: Window inclusion
          conditions.filter(c => ['accumulation_2pct_mc', 'delivery_times_3pct'].includes(c.condition)).forEach(c => {
            const start = normalizeDate(c.start_date);
            const end = normalizeDate(c.end_date);
            if (start && end && rowDate >= start && rowDate <= end) applicable.push(c.condition);
          });

          if (applicable.length > 0) {
            const priority = ['amount_1pct_mc', 'accumulation_2pct_mc', 'delivery_times_3pct'];
            const selected = priority.find(p => applicable.includes(p)) || applicable[0];
            console.log(`[DEBUG] ${symbol} Row ${R} (${rowDate}): ${selected} color`);
            const style = colorMap[selected];

            for (let C = range.s.c; C <= range.e.c; ++C) {
              const addr = XLSX.utils.encode_cell({ r: R, c: C });
              if (!worksheet[addr]) continue;
              worksheet[addr].s = { ... (worksheet[addr].s || {}), ...style };
            }
            coloredCount++;
          }
        }
        console.log(`[DEBUG] ${symbol}: ${coloredCount} rows colored`);

        // Usability
        worksheet['!freeze'] = 'A2';
        worksheet['!autofilter'] = { ref: worksheet['!ref'] };

        return worksheet;
      };

      const workbook = XLSX.utils.book_new();

      if (isMultiSymbol) {
        Object.entries(masterAnalysis).forEach(([sym, data]) => {
          const ws = processSymbolData(sym, data);
          if (ws) XLSX.utils.book_append_sheet(workbook, ws, `${sym}_Data`);
        });

        // Summary sheet (enhanced legend)
        const summaryData = [
          ['Master File Analysis - Multi-Symbol (Colors Applied)'],
          ['Symbols Analyzed', Object.keys(masterAnalysis).length],
          [''],  // Spacer
          // Table headers and data (adapt from your original summary logic)
          ['Symbol', 'Market Cap (Cr)', 'Conditions Met', 'Qualifying Dates', 'Delv >3%', 'Amount >1%', 'Accum >2%', 'Total >1%', 'Total Accum (Cr)', '% MC', 'Status'],
          // ... (populate with your data; omitted for brevity)
          [''],  // Spacer
          ['🗺️ Color Legend (Full Row Highlighting)'],
          ['ARGB Hex', 'Color/Emoji', 'Condition', 'Threshold & Meaning', 'Priority', 'When Rows Are Colored'],
          ['FFFFCDD2', '🟥 Red', 'Amount >1% MC', 'Daily amount >1% market cap (e.g., ₹100Cr for ₹10K Cr MC; signals major buy)', 'Highest', 'Exact matching date: Entire row red'],
          ['FFC8E6C9', '🟩 Green', 'Accum >2% MC', '15-day sum >2% MC (sustained institutional accumulation)', 'High', 'Dates in window: Entire row green'],
          ['FFFFE082', '🟨 Yellow', 'Delv Times >3%', '>3x avg delivery in 2-3 days (short-term strength)', 'Medium', 'Dates in window: Entire row yellow'],
          ['FFE1F5FE', '🔵 Light Blue', 'Total Accum >1% MC', 'Full period >1% MC (overall buying interest)', 'Base/Lowest', 'All rows blue if met (overridden by above)'],
          [''],  // Spacer
          ['Notes: Colors use ARGB for Excel compatibility. Check console for per-symbol counts. Data: Uploaded file + Yahoo Finance. Open in MS Excel for best view.']
        ];
        const summaryWs = XLSX.utils.aoa_to_sheet(summaryData);
        summaryWs['!freeze'] = 'A2';
        XLSX.utils.book_append_sheet(workbook, summaryWs, 'Overall_Summary');
      } else {
        // Single-symbol logic (mirror multi; populate summaryData similarly)
        const symData = Object.values(masterAnalysis)[0];
        const ws = processSymbolData(symData.symbol, symData);
        if (ws) XLSX.utils.book_append_sheet(workbook, ws, 'Highlighted_Data');

        // Single summary (adapt table/legend as above)
        const summaryData = [ /* ... similar to multi, but single row ... */ ];
        const summaryWs = XLSX.utils.aoa_to_sheet(summaryData);
        summaryWs['!freeze'] = 'A2';
        XLSX.utils.book_append_sheet(workbook, summaryWs, 'Analysis_Summary');
      }

      const timestamp = format(new Date(), 'yyyy-MM-dd');
      const filename = isMultiSymbol 
        ? `MasterAnalysis_MultiSymbol_Highlighted_${timestamp}.xlsx` 
        : `${Object.keys(masterAnalysis)[0] || 'Stock'}_Highlighted_${timestamp}.xlsx`;

      const buffer = XLSX.write(workbook, { bookType: 'xlsx', type: 'array', bookSST: false });
      const blob = new Blob([buffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
      saveAs(blob, filename);

      console.log(`[SUCCESS] Generated ${filename} – Check Excel for colors`);
      setShowModal(false);
    } catch (err) {
      console.error('[ERROR] Excel gen failed:', err);
      setError(`Generation failed: ${err.message}`);
    } finally {
      setDownloading(false);
    }
  };

  const getConditionSummary = () => {
    if (!analysisData?.master_analysis) return null;
    const masterAnalysis = analysisData.master_analysis;
    const isMulti = Object.keys(masterAnalysis).length > 1;
    if (isMulti) {
      const totals = Object.values(masterAnalysis).reduce((acc, data) => ({
        conditions: acc.conditions + (data.summary_stats?.total_conditions_met || 0),
        dates: acc.dates + (data.summary_stats?.qualifying_dates_count || 0),
        accMet: acc.accMet + (data.summary_stats?.condition4_count || 0),
        accAmount: acc.accAmount + (data.detailed_analysis?.total_accumulation_analysis?.total_amount || 0),
        accPct: acc.accPct + (data.detailed_analysis?.total_accumulation_analysis?.total_percentage || 0)
      }), { conditions: 0, dates: 0, accMet: 0, accAmount: 0, accPct: 0 });
      const avgPct = totals.accPct / Object.keys(masterAnalysis).length;
      return (
        <div className="small">
          <div className="mb-2"><strong>Symbols:</strong> {Object.keys(masterAnalysis).length}</div>
          <div className="mb-2"><strong>Total Conditions:</strong> {totals.conditions}</div>
          <div className="mb-2"><strong>Total Dates:</strong> {totals.dates}</div>
          <div className="mb-2"><strong>Accum {'>'} 1% Met:</strong> {totals.accMet}/{Object.keys(masterAnalysis).length}</div>
          <div><strong>Avg Accum:</strong> ₹{Number(totals.accAmount / 1e7).toLocaleString('en-IN', { maximumFractionDigits: 2 })} ({avgPct.toFixed(2)}% MC)</div>
        </div>
      );
    } else {
      const data = Object.values(masterAnalysis)[0];
      const stats = data.summary_stats || {};
      const acc = data.detailed_analysis?.total_accumulation_analysis || {};
      return (
        <div className="small">
          <div className="mb-2"><strong>Conditions Met:</strong> {stats.total_conditions_met}</div>
          <div className="mb-2">Delv: {stats.condition1_count} | Amount: {stats.condition2_count} | Accum: {stats.condition3_count} | Total: {stats.condition4_count} ({acc.total_percentage?.toFixed(2)}% MC – {stats.condition4_count > 0 ? 'Met' : 'Not'})</div>
          <div><strong>Dates:</strong> {stats.qualifying_dates_count}</div>
        </div>
      );
    }
  };

  if (!analysisData?.master_analysis) return null;

  return (
    <>
      <Button variant="success" onClick={() => setShowModal(true)} className="mb-3" size="sm">
        <i className="fas fa-download me-2"></i>Download Highlighted Excel
      </Button>
      <Modal show={showModal} onHide={() => setShowModal(false)} size="lg">
        <Modal.Header closeButton>
          <Modal.Title><i className="fas fa-file-excel me-2 text-success"></i>Download Master Analysis Excel (Colors Enabled)</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {error && <Alert variant="danger" dismissible onClose={() => setError(null)}>{error}</Alert>}
          <div className="mb-4">
            <h6 className="fw-bold text-primary mb-3">Summary</h6>
            {getConditionSummary()}
          </div>
          <div className="mb-4">
            <h6 className="fw-bold text-primary mb-3">Color System (Row Highlighting)</h6>
            <div className="d-flex flex-wrap gap-3">
              {[
                { hex: 'FFFFCDD2', emoji: '🟥', cond: 'Amount >1% MC', desc: 'Single-day >1% MC: Highest, full row red' },
                { hex: 'FFC8E6C9', emoji: '🟩', cond: 'Accum >2% MC', desc: '15-day window: Full row green' },
                { hex: 'FFFFE082', emoji: '🟨', cond: 'Delv >3%', desc: '2-3 day window: Full row yellow' },
                { hex: 'FFE1F5FE', emoji: '🔵', cond: 'Total >1% MC', desc: 'All rows blue base (overridden)' }
              ].map((c, i) => (
                <div key={i} className="d-flex align-items-center border p-2 rounded bg-light">
                  <div className="me-2" style={{ width: '20px', height: '20px', backgroundColor: `#${c.hex.slice(2)}`, border: '1px solid #ccc' }}></div>
                  <small><strong>{c.emoji} {c.cond}</strong><br/><em>{c.desc}</em></small>
                </div>
              ))}
            </div>
          </div>
          <Alert variant="info">
            <Alert.Heading>Included</Alert.Heading>
            <ul className="mb-0 small">
              <li>Highlighted sheets per symbol (multi) or single.</li>
              <li>Summary with metrics, legend (ARGB hexes), notes.</li>
              <li>Frozen headers, filters for easy navigation.</li>
              <li>Console logs row counts for verification.</li>
            </ul>
          </Alert>
          <div className="text-center">
            {downloading ? (
              <div className="d-flex align-items-center justify-content-center">
                <Spinner animation="border" size="sm" className="me-2" />
                <span>Generating...</span>
              </div>
            ) : (
              <Button variant="success" size="lg" onClick={downloadHighlightedExcel} className="px-4 py-2">
                <i className="fas fa-download me-2"></i>Download (Colors Applied)
              </Button>
            )}
          </div>
        </Modal.Body>
      </Modal>
    </>
  );
};

export default FileDownloadHandler;