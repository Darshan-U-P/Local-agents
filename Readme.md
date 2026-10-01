

````
# Local AI PPT Maker

A local-first AI presentation generator inspired by Gamma.

## Architecture

- React + Vite frontend
- Electron desktop application
- Python + FastAPI backend
- llama.cpp local inference
- Qwen3-4B for content and presentation planning
- Presentation Planner
- Presentation IR
- Layout Engine
- Asset Router
- Local FLUX image generation
- GPU + RAM + SSD model offloading
- Editable PowerPoint generation
- AI-powered slide editing

## Core Pipeline

User Prompt
    ↓
Qwen3-4B
    ↓
Presentation Planner
    ↓
Presentation IR
    ↓
Layout Engine
    ↓
Asset Router
    ↓
┌──────────┬──────────┬──────────┬──────────┐
│  Images  │  Icons   │ Diagrams │ Charts   │
│  FLUX    │  Local   │  Local   │  Local   │
└──────────┴──────────┴──────────┴──────────┘
    ↓
PPTX Renderer
    ↓
Editable PowerPoint

## Goals

Generate editable PowerPoint presentations from natural-language prompts.

The system is designed to run locally with minimal dependence on cloud AI services.

## Current Status

### Phase 1 — Local Qwen3-4B Inference
- Completed
- llama.cpp integration
- GGUF model support
- CUDA GPU offloading
- Local Qwen3-4B inference

### Phase 2 — Presentation Planner
- Completed
- Natural-language topic → presentation plan
- JSON-based planning
- Slide structure generation
- Layout selection
- Asset requirement generation

### Phase 3 — Presentation IR
- Completed
- Presentation intermediate representation
- Theme representation
- Slide representation
- Element representation
- Asset representation
- IR validation

### Phase 4 — Layout Engine
- Completed
- 16 slide layouts
- Element positioning
- Geometry validation
- Collision detection
- Slide boundary validation

### Phase 5 — Editable PPTX Renderer
- Completed
- Python-pptx renderer
- 16:9 presentation generation
- Editable text
- Editable PowerPoint shapes
- Image asset support
- PPTX validation

### Phase 6 — Asset Routing & Local Image Generation
- Completed
- Asset Router
- Image asset routing
- Icon asset routing
- Diagram asset routing
- Chart asset routing
- Local FLUX Q2_K GGUF integration
- CPU inference validation
- CUDA inference validation
- GPU + RAM offloading
- GPU + RAM + SSD offloading
- 512×512 generation
- 768×768 generation
- 1024×1024 generation
- Temporary SSD offload cache
- GPU memory cleanup

### Phase 7 — Production Asset Engine
- Next
- Production FLUX image generator
- Image model manager
- Asset caching
- Dynamic image resolution
- Asset Router integration
- Presentation IR asset integration

### Phase 8 — Icons, Diagrams & Charts
- Planned
- Local icon generation
- Editable diagrams
- Editable charts
- Structured visual assets

### Phase 9 — Full Presentation Generation Pipeline
- Planned
- Planner → IR → Layout → Assets → PPTX
- End-to-end presentation generation
- Automated asset generation
- Presentation validation

### Phase 10 — Quality Control
- Planned
- Layout validation
- Asset validation
- PPTX validation
- Rendering verification
- Automatic error detection

### Phase 11 — React/Vite UI
- Planned
- Presentation workspace
- Prompt interface
- Slide preview
- Generation progress
- Asset management

### Phase 12 — AI Slide Editor
- Planned
- AI-powered slide editing
- Regenerate slide
- Rewrite content
- Change layout
- Replace image
- Modify theme

### Phase 13 — Themes & Templates
- Planned
- Presentation themes
- Typography
- Color systems
- Slide templates
- Custom branding

### Phase 14 — Research & Web Grounding
- Planned
- Web research
- Source collection
- Fact grounding
- Citations
- Research-assisted presentations

### Phase 15 — Project Management & Export
- Planned
- Presentation projects
- Save/load projects
- Export PPTX
- Asset management
- Presentation history

## Local AI Models

### Language Model

- Qwen3-4B
- GGUF
- llama.cpp
- Local inference

### Image Model

- FLUX.1-schnell
- Q2_K GGUF
- Diffusers
- CUDA
- GPU/RAM/SSD offloading

## Hardware Target

The current development system is designed around a low-VRAM local AI setup.

Tested image generation on:

- NVIDIA RTX 3050 Ti Laptop GPU
- 4 GB VRAM
- CUDA
- System RAM offloading
- SSD offloading

Tested successfully up to:

- 1024 × 1024 image generation
- 4 inference steps

## Design Principles

- Local-first
- Editable output
- Modular architecture
- Deterministic presentation rendering
- Structured intermediate representation
- Hardware-aware AI inference
- Efficient memory utilization
- Minimal cloud dependency

## Repository Structure

```text
Local-agents/
│
├── backend/
│   ├── ai/
│   ├── planner/
│   ├── ir/
│   ├── layout/
│   ├── assets/
│   │   ├── generators/
│   │   └── asset_router.py
│   └── renderers/
│
├── config/
│
├── generated/
│
└── README.md
````

## Development Roadmap

```
Phase 1   Local Qwen3-4B Inference          ✅
Phase 2   Presentation Planner              ✅
Phase 3   Presentation IR                   ✅
Phase 4   Layout Engine                     ✅
Phase 5   Editable PPTX Renderer            ✅
Phase 6   Asset Routing + FLUX              ✅
Phase 7   Production Asset Engine           ▶
Phase 8   Icons / Diagrams / Charts         ○
Phase 9   Full Generation Pipeline          ○
Phase 10  Quality Control                   ○
Phase 11  React/Vite UI                     ○
Phase 12  AI Slide Editor                   ○
Phase 13  Themes / Templates                ○
Phase 14  Research / Web Grounding          ○
Phase 15  Project Management / Export       ○
```