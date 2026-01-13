import React, { useState, useEffect, useRef } from 'react';

const SEQ_LEN = 128;     // N - sequence length
const HEAD_DIM = 4;      // d - head dimension
const HBM_CAPACITY = 40000; // elements - holds Q, K, V, S (in naive), and O
const SRAM_CAPACITY = HBM_CAPACITY * 0.1; // SRAM is much smaller than HBM

const FlashAttentionViz = () => {
  const [tileSize, setTileSize] = useState(8);
  const [isRunning, setIsRunning] = useState(false);
  const [mode, setMode] = useState('flash'); // 'flash' or 'naive'
  const [speed, setSpeed] = useState(300);
  
  // State for tracking progress
  const [qTileIdx, setQTileIdx] = useState(0);
  const [kvTileIdx, setKvTileIdx] = useState(0);
  const [phase, setPhase] = useState('idle'); // 'idle', 'compute_s', 'accumulate_o', 'write_o'
  const [completedOTiles, setCompletedOTiles] = useState([]); // tiles written to HBM
  const [showSFlash, setShowSFlash] = useState(false);
  
  // Stats
  const [stats, setStats] = useState({ hbmReads: 0, hbmWrites: 0 });
  
  const intervalRef = useRef(null);
  const tilesPerDim = Math.ceil(SEQ_LEN / tileSize); // tiles along sequence dimension
  const totalQTiles = tilesPerDim;
  const totalKVTiles = tilesPerDim;

  // Calculate SRAM needed for a given tile size
  const getSramNeededForTileSize = (ts) => {
    // Q tile: ts × d, K tile: ts × d, V tile: ts × d, S tile: ts × ts, O accumulator: ts × d
    return ts * HEAD_DIM * 4 + ts * ts;
  };
  
  const sramNeededForCurrentTile = getSramNeededForTileSize(tileSize);
  const tilesFitInSram = sramNeededForCurrentTile <= SRAM_CAPACITY;

  // Calculate SRAM usage based on current phase
  const getSramContents = () => {
    if (phase === 'idle') return { q: 0, k: 0, v: 0, s: 0, o: 0 };
    
    // Q, K, V, O tiles are tileSize × HEAD_DIM
    // S tile is tileSize × tileSize (the N×N portion we're computing)
    const qkvTileElements = tileSize * HEAD_DIM;
    const sTileElements = tileSize * tileSize;
    const oTileElements = tileSize * HEAD_DIM;
    
    if (mode === 'naive') {
      // Naive: loads entire matrices, stores full S
      return {
        q: SEQ_LEN * HEAD_DIM,
        k: SEQ_LEN * HEAD_DIM,
        v: phase === 'accumulate_o' ? SEQ_LEN * HEAD_DIM : 0,
        s: SEQ_LEN * SEQ_LEN, // This is the killer - N×N!
        o: phase === 'accumulate_o' ? SEQ_LEN * HEAD_DIM : 0
      };
    }
    
    // Flash attention - only tiles in SRAM
    return {
      q: qkvTileElements,
      k: qkvTileElements,
      v: phase === 'accumulate_o' ? qkvTileElements : 0,
      s: showSFlash ? sTileElements : 0, // ephemeral
      o: oTileElements // accumulator always present
    };
  };

  const sramContents = getSramContents();
  const totalSram = sramContents.q + sramContents.k + sramContents.v + sramContents.s + sramContents.o;
  const sramOverflow = totalSram > SRAM_CAPACITY;

  const reset = () => {
    setIsRunning(false);
    setQTileIdx(0);
    setKvTileIdx(0);
    setPhase('idle');
    setCompletedOTiles([]);
    setShowSFlash(false);
    setStats({ hbmReads: 0, hbmWrites: 0 });
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
  };

  const advanceStep = () => {
    if (mode === 'naive') {
      // Naive has just two steps: compute S, then compute O
      if (phase === 'idle') {
        setPhase('compute_s');
        // Read Q and K (each N×d), write S (N×N)
        setStats({ 
          hbmReads: SEQ_LEN * HEAD_DIM * 2, 
          hbmWrites: SEQ_LEN * SEQ_LEN 
        });
      } else if (phase === 'compute_s') {
        setPhase('accumulate_o');
        // Additionally read S (N×N) and V (N×d), write O (N×d)
        setStats({ 
          hbmReads: SEQ_LEN * HEAD_DIM * 2 + SEQ_LEN * SEQ_LEN + SEQ_LEN * HEAD_DIM,
          hbmWrites: SEQ_LEN * SEQ_LEN + SEQ_LEN * HEAD_DIM
        });
      } else if (phase === 'accumulate_o') {
        // Mark all O tiles as complete
        const allTiles = [];
        for (let i = 0; i < tilesPerDim; i++) {
          allTiles.push(i);
        }
        setCompletedOTiles(allTiles);
        setPhase('idle');
        setIsRunning(false);
      }
      return;
    }
    
    // Flash Attention logic
    const qkvTileSize = tileSize * HEAD_DIM;
    const oTileSize = tileSize * HEAD_DIM;
    
    if (phase === 'idle') {
      // Start: load first Q tile, first K tile
      setPhase('compute_s');
      setShowSFlash(true);
      setStats({ hbmReads: qkvTileSize * 2, hbmWrites: 0 }); // Q tile + K tile
      return;
    }
    
    if (phase === 'compute_s') {
      // S computed, now load V and accumulate
      setPhase('accumulate_o');
      setShowSFlash(false); // S is ephemeral, used and discarded
      setStats(prev => ({ 
        ...prev, 
        hbmReads: prev.hbmReads + qkvTileSize // V tile
      }));
      return;
    }
    
    if (phase === 'accumulate_o') {
      // Move to next KV tile or next Q tile
      const nextKV = kvTileIdx + 1;
      
      if (nextKV < totalKVTiles) {
        // More KV tiles for this Q tile
        setKvTileIdx(nextKV);
        setPhase('compute_s');
        setShowSFlash(true);
        setStats(prev => ({ 
          ...prev, 
          hbmReads: prev.hbmReads + qkvTileSize // next K tile
        }));
      } else {
        // Done with this Q tile, write O to HBM
        setPhase('write_o');
        setStats(prev => ({ 
          ...prev, 
          hbmWrites: prev.hbmWrites + oTileSize // O tile written
        }));
      }
      return;
    }
    
    if (phase === 'write_o') {
      // Mark this O tile as complete
      setCompletedOTiles(prev => [...prev, qTileIdx]);
      
      const nextQ = qTileIdx + 1;
      if (nextQ < totalQTiles) {
        // More Q tiles to process
        setQTileIdx(nextQ);
        setKvTileIdx(0);
        setPhase('compute_s');
        setShowSFlash(true);
        setStats(prev => ({ 
          ...prev, 
          hbmReads: prev.hbmReads + qkvTileSize * 2 // new Q tile + first K tile
        }));
      } else {
        // All done
        setPhase('idle');
        setIsRunning(false);
      }
      return;
    }
  };

  useEffect(() => {
    if (isRunning) {
      intervalRef.current = setInterval(advanceStep, speed);
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isRunning, phase, qTileIdx, kvTileIdx, mode, tileSize, speed]);

  // Determine highlighting for each matrix
  // Q, V, O are N×d (SEQ_LEN × HEAD_DIM)
  // K^T is d×N (HEAD_DIM × SEQ_LEN) for display
  // S is N×N (SEQ_LEN × SEQ_LEN)
  const getHighlight = (matrix, row, col) => {
    if (phase === 'idle') return 'none';
    
    const qStart = qTileIdx * tileSize;
    const qEnd = Math.min(qStart + tileSize, SEQ_LEN);
    const kvStart = kvTileIdx * tileSize;
    const kvEnd = Math.min(kvStart + tileSize, SEQ_LEN);
    
    if (mode === 'naive') {
      // Naive highlights everything
      if (phase === 'compute_s' && (matrix === 'Q' || matrix === 'K')) return 'active';
      if (phase === 'accumulate_o') {
        if (matrix === 'S' || matrix === 'V') return 'active';
        if (matrix === 'O') return 'writing';
      }
      return 'none';
    }
    
    // Flash Attention
    // Q is N×d, we tile along rows (sequence dimension)
    if (matrix === 'Q') {
      if (row >= qStart && row < qEnd) return 'active';
    }
    
    // K^T is displayed as d×N, we tile along columns (sequence dimension)
    if (matrix === 'K') {
      if (col >= kvStart && col < kvEnd) return 'active';
    }
    
    // V is N×d, we tile along rows (sequence dimension)
    if (matrix === 'V') {
      if (phase === 'accumulate_o' && row >= kvStart && row < kvEnd) return 'active';
    }
    
    // S is N×N - the tile being computed is [qStart:qEnd, kvStart:kvEnd]
    if (matrix === 'S') {
      if (showSFlash && row >= qStart && row < qEnd && col >= kvStart && col < kvEnd) {
        return 'ephemeral';
      }
    }
    
    // O is N×d, we tile along rows (sequence dimension)
    if (matrix === 'O') {
      // Show completed tiles
      for (const completedQ of completedOTiles) {
        const cStart = completedQ * tileSize;
        const cEnd = Math.min(cStart + tileSize, SEQ_LEN);
        if (row >= cStart && row < cEnd) return 'complete';
      }
      // Show currently accumulating tile
      if (row >= qStart && row < qEnd && phase !== 'idle') {
        return 'accumulating';
      }
    }
    
    return 'none';
  };

  const getCellColor = (matrix, row, col) => {
    const highlight = getHighlight(matrix, row, col);
    
    switch (highlight) {
      case 'active':
        if (matrix === 'Q') return 'bg-blue-400';
        if (matrix === 'K') return 'bg-pink-400';
        if (matrix === 'V') return 'bg-purple-400';
        if (matrix === 'S') return 'bg-amber-400';
        return 'bg-gray-400';
      case 'ephemeral':
        return 'bg-amber-400 animate-pulse';
      case 'accumulating':
        return 'bg-green-300'; // lighter green - in progress
      case 'complete':
        return 'bg-green-500'; // solid green - written to HBM
      case 'writing':
        return 'bg-green-400';
      default:
        return 'bg-gray-200';
    }
  };

  const getIndicatorColor = (name) => {
    if (name === 'Q') return 'bg-blue-400';
    if (name === 'K') return 'bg-pink-400';
    if (name === 'V') return 'bg-purple-400';
    if (name === 'S') return 'bg-amber-400';
    if (name === 'O') return 'bg-green-400';
    return 'bg-gray-400';
  };

  const Matrix = ({ name, label, rows, cols }) => {
    // Smaller cells to fit larger matrices
    const cellSize = Math.max(2, Math.min(5, Math.floor(300 / Math.max(rows, cols))));
    
    return (
      <div className="flex flex-col items-center">
        <div className="text-xs font-semibold mb-1 flex items-center gap-1">
          <div className={`w-2 h-2 rounded ${getIndicatorColor(name)}`} />
          {label}
        </div>
        <div className="text-xs text-gray-500 mb-1">{rows}×{cols}</div>
        <div 
          className="border border-gray-400"
          style={{ 
            display: 'grid',
            gridTemplateColumns: `repeat(${cols}, ${cellSize}px)`,
            gap: '0px',
            backgroundColor: '#ccc'
          }}
        >
          {Array.from({ length: rows * cols }).map((_, idx) => {
            const row = Math.floor(idx / cols);
            const col = idx % cols;
            return (
              <div
                key={idx}
                className={`${getCellColor(name, row, col)} transition-colors duration-150`}
                style={{ width: cellSize, height: cellSize }}
              />
            );
          })}
        </div>
      </div>
    );
  };

  const getPhaseDescription = () => {
    if (phase === 'idle') return 'Ready to start';
    
    if (mode === 'naive') {
      if (phase === 'compute_s') return 'Computing S = Q × Kᵀ (loading ALL of Q and K, writing S to HBM)';
      if (phase === 'accumulate_o') return 'Computing O = softmax(S) × V (reading S back from HBM)';
    }
    
    const qTile = qTileIdx + 1;
    const kvTile = kvTileIdx + 1;
    
    if (phase === 'compute_s') {
      return `Q tile ${qTile}/${totalQTiles}: Computing S tile [${qTile},${kvTile}] — ephemeral, in SRAM only`;
    }
    if (phase === 'accumulate_o') {
      return `Q tile ${qTile}/${totalQTiles}: Accumulating into O (processed ${kvTile}/${totalKVTiles} KV tiles)`;
    }
    if (phase === 'write_o') {
      return `Writing completed O tile ${qTile} to HBM`;
    }
    return '';
  };

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <h1 className="text-2xl font-bold mb-2">Flash Attention Memory Patterns</h1>
      <p className="text-gray-600 mb-2">
        See how Flash Attention keeps intermediate results in fast SRAM and only writes final outputs to HBM.
      </p>
      
      {/* Math reminder */}
      <div className="mb-4 p-3 bg-gray-100 rounded-lg">
        <div className="text-center font-mono">
          <span className="text-green-700 font-bold">O</span>
          <span className="text-gray-600"> = softmax(</span>
          <span className="text-blue-600 font-bold">Q</span>
          <span className="text-pink-600 font-bold">K</span>
          <span className="text-gray-600"><sup>T</sup>) × </span>
          <span className="text-purple-600 font-bold">V</span>
        </div>
        <div className="text-center text-xs text-gray-500 mt-1">
          where <span className="text-amber-600 font-bold">S</span> = <span className="text-blue-600">Q</span><span className="text-pink-600">K</span><sup>T</sup> is the N×N attention scores matrix — <em>this is what Flash Attention avoids storing in HBM</em>
        </div>
      </div>
      
      {/* Tile size vs SRAM explanation */}
      {mode === 'flash' && (
        <div className="mb-4 p-3 bg-blue-50 rounded-lg text-sm">
          <div className="font-medium mb-2">Tile size is determined by SRAM capacity, not by d:</div>
          <div className="grid grid-cols-4 gap-2 text-xs">
            {[4, 8, 16, 32].map(ts => {
              const sramNeeded = ts * HEAD_DIM * 4 + ts * ts;
              const fits = sramNeeded <= SRAM_CAPACITY;
              const isSelected = ts === tileSize;
              return (
                <div 
                  key={ts} 
                  className={`p-2 rounded border ${isSelected ? 'border-blue-500 bg-blue-100' : 'border-gray-300'} ${!fits ? 'opacity-50' : ''}`}
                >
                  <div className="font-medium">{ts}×{ts} tiles</div>
                  <div>SRAM: {sramNeeded.toLocaleString()}</div>
                  <div>K/V re-reads: {Math.ceil(SEQ_LEN/ts)}×</div>
                  <div className={fits ? 'text-green-600' : 'text-red-600'}>
                    {fits ? '✓ fits' : '✗ too large'}
                  </div>
                </div>
              );
            })}
          </div>
          <div className="mt-2 text-gray-600">
            → Use the <strong>largest tile that fits in SRAM</strong> to minimize K/V re-reads.
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="flex flex-wrap gap-4 mb-6 p-4 bg-gray-100 rounded-lg">
        <div className="flex items-center gap-2">
          <label className="font-medium">Mode:</label>
          <select 
            value={mode === 'naive' ? 'naive' : tileSize.toString()}
            onChange={(e) => {
              reset();
              if (e.target.value === 'naive') {
                setMode('naive');
              } else {
                setMode('flash');
                setTileSize(parseInt(e.target.value));
              }
            }}
            className="border rounded px-2 py-1"
          >
            <option value="naive">Naive (no tiling)</option>
            <option value="4">4×4 tiles ({Math.ceil(SEQ_LEN/4)}× re-reads)</option>
            <option value="8">8×8 tiles ({Math.ceil(SEQ_LEN/8)}× re-reads)</option>
            <option value="16">16×16 tiles ({Math.ceil(SEQ_LEN/16)}× re-reads)</option>
            <option value="32">32×32 tiles ({Math.ceil(SEQ_LEN/32)}× re-reads)</option>
          </select>
        </div>
        
        <div className="flex items-center gap-2">
          <label className="font-medium">Speed:</label>
          <input
            type="range"
            min="50"
            max="500"
            value={500 - speed + 50}
            onChange={(e) => setSpeed(500 - parseInt(e.target.value) + 50)}
            className="w-24"
          />
        </div>
        
        <button
          onClick={() => isRunning ? setIsRunning(false) : setIsRunning(true)}
          className="px-4 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          {isRunning ? 'Pause' : (phase === 'idle' && completedOTiles.length === 0 ? 'Start' : 'Resume')}
        </button>
        
        <button
          onClick={reset}
          className="px-4 py-1 bg-gray-500 text-white rounded hover:bg-gray-600"
        >
          Reset
        </button>
      </div>

      {/* HBM Stats */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="p-4 bg-blue-50 rounded-lg">
          <div className="text-sm text-gray-600">HBM Reads</div>
          <div className="text-2xl font-bold text-blue-700">
            {stats.hbmReads.toLocaleString()}
          </div>
          <div className="text-xs text-gray-500">elements loaded from slow memory</div>
        </div>
        <div className="p-4 bg-green-50 rounded-lg">
          <div className="text-sm text-gray-600">HBM Writes</div>
          <div className="text-2xl font-bold text-green-700">
            {stats.hbmWrites.toLocaleString()}
          </div>
          <div className="text-xs text-gray-500">elements written to slow memory</div>
        </div>
        <div className="p-4 bg-purple-50 rounded-lg">
          <div className="text-sm text-gray-600">Total HBM Traffic</div>
          <div className="text-2xl font-bold text-purple-700">
            {(stats.hbmReads + stats.hbmWrites).toLocaleString()}
          </div>
          <div className="text-xs text-gray-500">
            reads + writes (lower is better)
          </div>
        </div>
      </div>

      {/* Comparison summary - focus on key difference */}
      <div className="mb-6 p-3 bg-gray-50 rounded-lg text-xs">
        <div className="font-medium mb-2">Key difference:</div>
        <div className="grid grid-cols-2 gap-4">
          <div className="p-2 bg-red-50 rounded">
            <span className="font-medium text-red-700">Naive writes S to HBM:</span>{' '}
            {(SEQ_LEN * SEQ_LEN).toLocaleString()} elements written, then read back.
            <br />
            <span className="text-gray-600">Total S traffic: {(SEQ_LEN * SEQ_LEN * 2).toLocaleString()} elements</span>
          </div>
          <div className="p-2 bg-green-50 rounded">
            <span className="font-medium text-green-700">Flash never writes S:</span>{' '}
            S tiles computed in SRAM and discarded.
            <br />
            <span className="text-gray-600">S traffic: 0 elements (saved {(SEQ_LEN * SEQ_LEN * 2).toLocaleString()}!)</span>
          </div>
        </div>
      </div>

      {/* SRAM Visual */}
      <div className="mb-4 p-4 bg-gray-100 rounded-lg">
        <div className="flex justify-between items-center mb-2">
          <div className="font-medium">SRAM (Fast On-Chip Memory)</div>
          <div className="text-sm text-gray-600">
            {totalSram.toLocaleString()} / {SRAM_CAPACITY.toLocaleString()} elements
          </div>
        </div>
        
        {/* SRAM capacity bar - 10% width to show it's much smaller than HBM */}
        <div className="relative h-8 bg-gray-300 rounded overflow-visible" style={{ width: '10%' }}>
          {phase !== 'idle' && (
            <div className="absolute left-0 top-0 bottom-0 right-0 flex">
              {/* Q segment - blue */}
              {sramContents.q > 0 && (
                <div 
                  className="bg-blue-400 h-full transition-all duration-300"
                  style={{ width: `${(sramContents.q / SRAM_CAPACITY) * 100}%` }}
                />
              )}
              {/* K segment - pink */}
              {sramContents.k > 0 && (
                <div 
                  className="bg-pink-400 h-full transition-all duration-300"
                  style={{ width: `${(sramContents.k / SRAM_CAPACITY) * 100}%` }}
                />
              )}
              {/* V segment - purple */}
              {sramContents.v > 0 && (
                <div 
                  className="bg-purple-400 h-full transition-all duration-300"
                  style={{ width: `${(sramContents.v / SRAM_CAPACITY) * 100}%` }}
                />
              )}
              {/* S segment - amber (ephemeral) */}
              {sramContents.s > 0 && (
                <div 
                  className="bg-amber-400 h-full transition-all duration-300 animate-pulse"
                  style={{ width: `${(sramContents.s / SRAM_CAPACITY) * 100}%` }}
                />
              )}
              {/* O accumulator - green */}
              {sramContents.o > 0 && (
                <div 
                  className="bg-green-400 h-full transition-all duration-300"
                  style={{ width: `${(sramContents.o / SRAM_CAPACITY) * 100}%` }}
                />
              )}
            </div>
          )}
          
          {/* Overflow indicator */}
          {sramOverflow && (
            <div 
              className="absolute top-0 bottom-0 bg-red-300 border-l-2 border-red-600 flex items-center justify-center rounded-r"
              style={{ left: '100%', width: '60px' }}
            >
              <span className="text-xs font-bold text-red-800 px-1">SPILL!</span>
            </div>
          )}
          
          {/* Capacity line */}
          <div className="absolute right-0 top-0 bottom-0 w-1 bg-gray-700 rounded" />
        </div>
        
        {/* SRAM Legend */}
        {phase !== 'idle' && (
          <div className="mt-2 flex flex-wrap gap-3 text-xs">
            {sramContents.q > 0 && (
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 bg-blue-400 rounded" />
                <span>Q tile ({sramContents.q})</span>
              </div>
            )}
            {sramContents.k > 0 && (
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 bg-pink-400 rounded" />
                <span>K tile ({sramContents.k})</span>
              </div>
            )}
            {sramContents.v > 0 && (
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 bg-purple-400 rounded" />
                <span>V tile ({sramContents.v})</span>
              </div>
            )}
            {sramContents.s > 0 && (
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 bg-amber-400 rounded animate-pulse" />
                <span>S tile ({sramContents.s}) — ephemeral!</span>
              </div>
            )}
            {sramContents.o > 0 && (
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 bg-green-400 rounded" />
                <span>O accumulator ({sramContents.o})</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* HBM Visual */}
      <div className="mb-6 p-4 bg-gray-100 rounded-lg">
        <div className="flex justify-between items-center mb-2">
          <div className="font-medium">HBM (Slow GPU Memory)</div>
          <div className="text-sm text-gray-600">
            {HBM_CAPACITY.toLocaleString()} elements total capacity
          </div>
        </div>
        
        {/* HBM bar - full width, shows stored data scaled to capacity */}
        <div className="relative h-8 bg-gray-300 rounded overflow-visible" style={{ width: '100%' }}>
          <div className="absolute left-0 top-0 bottom-0 right-0 flex">
            {/* Q in HBM - N×d */}
            <div 
              className="bg-blue-400 h-full border-r border-blue-600"
              style={{ width: `${(SEQ_LEN * HEAD_DIM / HBM_CAPACITY) * 100}%` }}
            />
            {/* K in HBM - N×d */}
            <div 
              className="bg-pink-400 h-full border-r border-pink-600"
              style={{ width: `${(SEQ_LEN * HEAD_DIM / HBM_CAPACITY) * 100}%` }}
            />
            {/* V in HBM - N×d */}
            <div 
              className="bg-purple-400 h-full border-r border-purple-600"
              style={{ width: `${(SEQ_LEN * HEAD_DIM / HBM_CAPACITY) * 100}%` }}
            />
            {/* S in HBM - N×N - only in naive mode! This is the big one! */}
            {mode === 'naive' && phase !== 'idle' && (
              <div 
                className="bg-amber-400 h-full border-r border-amber-600"
                style={{ width: `${(SEQ_LEN * SEQ_LEN / HBM_CAPACITY) * 100}%` }}
              />
            )}
            {/* O in HBM - builds up as tiles complete (flash) or all at once (naive) */}
            {mode === 'flash' && completedOTiles.length > 0 && (
              <div 
                className="bg-green-500 h-full transition-all duration-300"
                style={{ width: `${((completedOTiles.length / tilesPerDim) * SEQ_LEN * HEAD_DIM / HBM_CAPACITY) * 100}%` }}
              />
            )}
            {mode === 'naive' && phase === 'idle' && completedOTiles.length > 0 && (
              <div 
                className="bg-green-500 h-full"
                style={{ width: `${(SEQ_LEN * HEAD_DIM / HBM_CAPACITY) * 100}%` }}
              />
            )}
          </div>
        </div>
        
        {/* HBM Legend */}
        <div className="mt-2 flex flex-wrap gap-3 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-blue-400 rounded" />
            <span>Q ({SEQ_LEN}×{HEAD_DIM}={SEQ_LEN * HEAD_DIM})</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-pink-400 rounded" />
            <span>K ({SEQ_LEN}×{HEAD_DIM}={SEQ_LEN * HEAD_DIM})</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-purple-400 rounded" />
            <span>V ({SEQ_LEN}×{HEAD_DIM}={SEQ_LEN * HEAD_DIM})</span>
          </div>
          {mode === 'naive' && phase !== 'idle' && (
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-amber-400 rounded" />
              <span className="text-red-600 font-medium">S ({SEQ_LEN}×{SEQ_LEN}={SEQ_LEN * SEQ_LEN}) — 4× larger!</span>
            </div>
          )}
          {(completedOTiles.length > 0 || (mode === 'naive' && phase === 'idle' && completedOTiles.length > 0)) && (
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-green-500 rounded" />
              <span>O ({SEQ_LEN}×{HEAD_DIM}) — {mode === 'flash' ? `${completedOTiles.length}/${tilesPerDim} tiles` : 'complete'}</span>
            </div>
          )}
        </div>
        
        {/* Key insight message */}
        <div className="mt-2 text-xs">
          {mode === 'naive' && phase !== 'idle' && (
            <span className="text-red-700 font-medium">
              ⚠️ S is {SEQ_LEN}×{SEQ_LEN} = {SEQ_LEN * SEQ_LEN} elements — 4× larger than Q, K, or V! This is what Flash Attention avoids.
            </span>
          )}
          {mode === 'flash' && phase !== 'idle' && (
            <span className="text-emerald-700">
              ✓ No S in HBM! Flash Attention only writes O tiles as they complete.
            </span>
          )}
          {phase === 'idle' && completedOTiles.length > 0 && mode === 'flash' && (
            <span className="text-emerald-700">
              ✓ Done! S was computed {tilesPerDim * tilesPerDim} times but never touched HBM.
            </span>
          )}
        </div>
      </div>

      {/* Phase indicator */}
      <div className="mb-4 p-3 bg-yellow-50 rounded-lg">
        <span className="font-medium">Current: </span>
        {getPhaseDescription()}
      </div>

      {/* Matrices */}
      <div className="flex flex-wrap justify-center items-end gap-3 mb-6">
        <Matrix name="Q" label="Q" rows={SEQ_LEN} cols={HEAD_DIM} />
        <div className="flex items-center text-xl pb-8">×</div>
        <Matrix name="K" label="Kᵀ" rows={HEAD_DIM} cols={SEQ_LEN} />
        <div className="flex items-center text-xl pb-8">=</div>
        <Matrix name="S" label="S" rows={SEQ_LEN} cols={SEQ_LEN} />
        <div className="flex items-center text-xl pb-8">×</div>
        <Matrix name="V" label="V" rows={SEQ_LEN} cols={HEAD_DIM} />
        <div className="flex items-center text-xl pb-8">=</div>
        <Matrix name="O" label="O" rows={SEQ_LEN} cols={HEAD_DIM} />
      </div>

      {/* Legend for O colors */}
      <div className="flex justify-center gap-6 mb-6 text-xs">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-green-300 border border-gray-400 rounded" />
          <span>O tile accumulating (in SRAM)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-green-500 border border-gray-400 rounded" />
          <span>O tile complete (written to HBM)</span>
        </div>
      </div>

      {/* Matrix dimension explanation */}
      <div className="p-4 bg-blue-50 rounded-lg text-sm mb-4">
        <h3 className="font-bold mb-2">Matrix Dimensions (N={SEQ_LEN}, d={HEAD_DIM}):</h3>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div><strong>Q, K, V, O:</strong> {SEQ_LEN}×{HEAD_DIM} = {SEQ_LEN * HEAD_DIM} elements each</div>
          <div><strong>S (scores):</strong> {SEQ_LEN}×{SEQ_LEN} = {SEQ_LEN * SEQ_LEN} elements — <span className="text-red-600 font-medium">4× larger!</span></div>
        </div>
        <p className="mt-2 text-gray-600">
          As sequence length N grows, S grows quadratically (N²) while Q, K, V, O only grow linearly (N×d). 
          This is why avoiding S storage matters so much for long sequences.
        </p>
      </div>

      {/* Explanation */}
      <div className="p-4 bg-gray-50 rounded-lg text-sm">
        <h3 className="font-bold mb-2">What you're seeing:</h3>
        {mode === 'naive' ? (
          <ul className="list-disc list-inside space-y-1">
            <li><strong>Naive attention</strong> computes the full S matrix ({SEQ_LEN}×{SEQ_LEN} = {SEQ_LEN * SEQ_LEN} elements) and writes it to HBM</li>
            <li>Then reads S back from HBM to multiply with V</li>
            <li>S alone is {Math.round(SEQ_LEN * SEQ_LEN / (SEQ_LEN * HEAD_DIM))}× larger than Q, K, V, or O!</li>
            <li>This causes massive memory traffic and can overflow SRAM</li>
          </ul>
        ) : (
          <ul className="list-disc list-inside space-y-1">
            <li><strong>Flash Attention</strong> processes one Q tile ({tileSize}×{HEAD_DIM}) at a time</li>
            <li>For each Q tile, it iterates through all K/V tiles</li>
            <li>S tiles ({tileSize}×{tileSize}) are <strong>ephemeral</strong> — computed, used, discarded (never written to HBM)</li>
            <li>O tiles accumulate in SRAM, only written to HBM when complete</li>
            <li>Total S tiles computed: {tilesPerDim}×{tilesPerDim} = {tilesPerDim * tilesPerDim}, but none stored!</li>
          </ul>
        )}
      </div>

      {/* Online softmax explanation - only show in flash mode */}
      {mode === 'flash' && (
        <div className="mt-4 p-4 bg-amber-50 rounded-lg text-sm border border-amber-200">
          <h3 className="font-bold mb-2">🔑 The Key Trick: Online Softmax</h3>
          <p className="mb-3 text-gray-700">
            But wait — softmax needs to see <em>all</em> scores to normalize. How can we compute O tile-by-tile without storing S?
          </p>
          
          <div className="mb-3 p-3 bg-white rounded border border-amber-100">
            <div className="font-medium mb-2">Standard softmax requires the full row:</div>
            <div className="font-mono text-xs text-center">
              softmax(s<sub>i</sub>) = exp(s<sub>i</sub>) / Σ exp(s<sub>j</sub>)
            </div>
            <div className="text-xs text-gray-500 text-center mt-1">
              Need the sum over <em>all</em> j to compute <em>any</em> output
            </div>
          </div>

          <div className="mb-3 p-3 bg-white rounded border border-amber-100">
            <div className="font-medium mb-2">Online softmax tracks running statistics:</div>
            <div className="grid grid-cols-3 gap-2 text-xs mb-2">
              <div className="p-2 bg-gray-50 rounded">
                <span className="font-mono font-bold">m</span> = running max
              </div>
              <div className="p-2 bg-gray-50 rounded">
                <span className="font-mono font-bold">ℓ</span> = Σ exp(s - m)
              </div>
              <div className="p-2 bg-gray-50 rounded">
                <span className="font-mono font-bold">acc</span> = unnormalized output
              </div>
            </div>
          </div>

          <div className="mb-3 p-3 bg-white rounded border border-amber-100">
            <div className="font-medium mb-2">When we see a new K/V tile:</div>
            <div className="font-mono text-xs space-y-1 bg-gray-900 text-gray-100 p-2 rounded">
              <div><span className="text-amber-400">s_tile</span> = q @ k_tile.T        <span className="text-gray-500"># compute local scores</span></div>
              <div><span className="text-amber-400">m_new</span> = max(m, max(s_tile))   <span className="text-gray-500"># update running max</span></div>
              <div><span className="text-amber-400">correction</span> = exp(m - m_new)  <span className="text-gray-500"># rescale factor</span></div>
              <div><span className="text-amber-400">ℓ</span> = ℓ * correction + Σ exp(s_tile - m_new)</div>
              <div><span className="text-amber-400">acc</span> = acc * correction + exp(s_tile - m_new) @ v_tile</div>
            </div>
          </div>

          <div className="p-3 bg-white rounded border border-amber-100">
            <div className="font-medium mb-2">At the end:</div>
            <div className="font-mono text-center text-sm">
              <span className="text-green-700 font-bold">O</span> = acc / ℓ
            </div>
            <div className="text-xs text-gray-500 text-center mt-1">
              One division normalizes everything — we never stored the softmax weights!
            </div>
          </div>

          <p className="mt-3 text-gray-600 text-xs">
            <strong>Why the correction works:</strong> When we discover a larger max, exp(old_max - new_max) scales down 
            all previous contributions, as if we'd known the true max from the start. The math is exact, not approximate.
          </p>
        </div>
      )}
    </div>
  );
};

export default FlashAttentionViz;
