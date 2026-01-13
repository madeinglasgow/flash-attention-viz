# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an interactive Flash Attention visualization built with React and Tailwind CSS. It demonstrates how Flash Attention reduces memory bandwidth compared to naive attention by avoiding materialization of the N×N attention scores matrix (S) in HBM.

## Development Commands

```bash
# From the flash-attention-demo directory (after setup):
npm install           # Install dependencies
npm run dev           # Start dev server at localhost:5173
npm run build         # Production build
```

## Setup

This project requires creating a new Vite + React project and adding the visualization component. See `flash_attention_viz_setup.md` for step-by-step instructions:

1. Create Vite React project
2. Install and configure Tailwind CSS
3. Copy `flash_attention_viz.jsx` to `src/FlashAttentionViz.jsx`
4. Update `App.jsx` to render the component

## Architecture

**Single-component visualization** (`flash_attention_viz.jsx`):
- Constants define matrix dimensions: `SEQ_LEN=128`, `HEAD_DIM=4`, `HBM_CAPACITY`, `SRAM_CAPACITY`
- State machine drives the animation with phases: `idle`, `compute_s`, `accumulate_o`, `write_o`
- Two modes: `flash` (tiled attention) and `naive` (standard attention)
- Tracks HBM reads/writes to demonstrate memory savings
- `Matrix` component renders each matrix (Q, K, V, S, O) with cell highlighting based on current tile

**Key visualization concepts**:
- Shows how Flash Attention keeps S tiles ephemeral (computed in SRAM, never written to HBM)
- Demonstrates online softmax algorithm that enables incremental O accumulation
- Tile size is constrained by SRAM capacity: `(tileSize × HEAD_DIM × 4) + (tileSize²) <= SRAM_CAPACITY`
