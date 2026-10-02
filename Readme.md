

````
# Local AI PPT Maker

A local-first AI presentation generator inspired by Gamma.

The system converts natural-language presentation requests into editable PowerPoint presentations using locally running AI models and deterministic rendering components.

---

## Architecture

- React + Vite frontend
- Electron desktop application
- Python + FastAPI backend
- llama.cpp local inference
- Qwen3-4B for content and presentation planning
- Presentation Planner
- Structured Visual Specifications
- Presentation IR
- Layout Engine
- Asset Prompt Builder
- Asset Router
- Local FLUX image generation
- GPU + RAM + SSD model offloading
- Editable PowerPoint generation
- AI-powered slide editing

---

## Core Pipeline

```text
User Prompt
    ↓
Qwen3-4B
    ↓
Presentation Planner
    ↓
Presentation Plan
    ↓
Structured Visual Specifications
    ↓
Presentation IR
    ↓
Layout Engine
    ↓
Asset Prompt Builder
    ↓
Asset Router
    ↓
┌──────────────┬──────────────┬──────────────┬──────────────┐
│    Images    │    Icons     │   Diagrams   │    Charts    │
│    FLUX      │    Local     │    Local     │    Local     │
│              │   Generator  │   Generator  │   Generator  │
└──────────────┴──────────────┴──────────────┴──────────────┘
    ↓
PPTX Renderer
    ↓
Editable PowerPoint
````

---

## Goals

Generate editable PowerPoint presentations from natural-language prompts.

The system is designed to run locally with minimal dependence on cloud AI services.

The long-term goal is to provide a local AI presentation workflow capable of:

- Presentation planning
- Content generation
- Visual asset planning
- Image generation
- Icon generation
- Diagram generation
- Chart generation
- Automatic layout
- Editable PowerPoint rendering
- AI-powered slide editing

---

# Current Status

## Phase 1 — Local Qwen3-4B Inference

**Completed**

- llama.cpp integration
- GGUF model support
- CUDA GPU offloading
- Local Qwen3-4B inference
- Lazy model loading
- Explicit model unloading
- GPU/CPU resource cleanup

---

## Phase 2 — Presentation Planner

**Completed**

- Natural-language topic → presentation plan
- JSON-based planning
- Slide structure generation
- Layout selection
- Asset requirement generation
- Asset type selection
- Structured visual specifications
- Visual purpose definition
- Subject definition
- Composition definition
- Style definition
- Color palette definition
- Required visual elements
- Negative visual constraints
- Text policy

The planner now separates **what a slide should communicate** from **how a visual asset should be generated**.

Example:

```
Presentation topic
        ↓
Slide purpose
        ↓
Asset requirement
        ↓
Visual specification
        ↓
Asset generator
```

---

## Phase 3 — Presentation IR

**Completed**

- Presentation intermediate representation
- Theme representation
- Slide representation
- Element representation
- Asset representation
- IR validation
- Slide numbering validation
- Asset reference validation

The Presentation IR acts as the structured contract between planning, layout, asset generation, and rendering.

---

## Phase 4 — Layout Engine

**Completed**

- 16 slide layouts
- Element positioning
- Geometry validation
- Collision detection
- Slide boundary validation
- 16:9 slide geometry
- Text positioning
- Asset positioning

The layout engine produces deterministic slide geometry instead of asking the language model to directly position PowerPoint elements.

---

## Phase 5 — Editable PPTX Renderer

**Completed**

- Python-pptx renderer
- 16:9 presentation generation
- Editable text
- Editable PowerPoint shapes
- Image asset support
- PPTX validation
- Placeholder support for unavailable assets

The renderer converts the structured presentation representation into an editable `.pptx` file.

---

# Phase 6 — Asset Routing & Local Image Generation

**Completed**

- Asset Router
- Image asset routing
- Icon asset routing
- Diagram asset routing
- Chart asset routing
- Local FLUX Q2_K GGUF integration
- Diffusers integration
- CPU inference validation
- CUDA inference validation
- GPU + RAM offloading
- GPU + RAM + SSD offloading
- 512×512 generation
- 768×768 generation
- 1024×1024 generation
- Temporary SSD offload cache
- GPU memory cleanup
- Batch FLUX model lifecycle
- Explicit FLUX unloading
- Resumable asset generation

### FLUX Model

```
FLUX.1-schnell
Q2_K GGUF
        ↓
Diffusers
        ↓
CUDA
        ↓
GPU + RAM + SSD offloading
```

The FLUX model is loaded once for an image-generation batch and unloaded after the batch completes.

---

# Phase 7 — Production Asset Engine

**In Progress**

### Completed

- Production FLUX image generator
- Image model manager
- Asset generation manager
- Asset generation sessions
- Asset manifest
- Presentation plan manifest
- Asset caching infrastructure
- Batch image generation
- Dynamic asset routing
- Image / illustration / photo asset support
- Structured visual-spec propagation
- Asset Prompt Builder
- Detailed visual prompt construction
- FLUX generation metadata tracking
- Generation seed tracking
- Resolution tracking
- Inference-step tracking
- Model/runtime tracking
- Generation failure tracking
- Resumable generation

### Current Work

- FLUX prompt optimization
- CLIP prompt-length optimization
- Visual-spec compression
- Improved image-generation accuracy
- Asset validation
- Better asset-to-slide integration

### Asset Generation Flow

```
Presentation Plan
        ↓
Asset Manifest
        ↓
AssetGenerationManager
        ↓
AssetRouter
        ↓
AssetPromptBuilder
        ↓
Compact generator prompt
        ↓
FLUX
        ↓
Generated image
        ↓
Asset Manifest
```

The full visual specification is preserved in the manifest while the generator receives a compact prompt optimized for the model's text encoder.

---

# Phase 8 — Icons, Diagrams & Charts

**Planned**

### Icons

- Local icon generation
- SVG/vector icon generation
- Structured icon specifications
- Editable asset integration

### Diagrams

- Structured diagram generation
- SVG/vector diagrams
- Editable PowerPoint diagrams
- Process diagrams
- Architecture diagrams
- Flow diagrams
- Relationship diagrams

### Charts

- Deterministic chart generation
- Structured chart specifications
- Editable PowerPoint charts
- Bar charts
- Line charts
- Pie charts
- Comparison charts
- Data validation

Charts and diagrams should preferably use deterministic/vector generation rather than image generation when editability and numerical accuracy are required.

---

# Phase 9 — Full Presentation Generation Pipeline

**Planned**

- Planner → IR → Layout → Assets → PPTX
- End-to-end presentation generation
- Automated asset generation
- Presentation validation
- Automatic session management
- Asset dependency resolution
- Complete presentation rendering

Target pipeline:

```
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
Asset Generation
    ↓
PPTX Renderer
    ↓
Validation
    ↓
Final .pptx
```

---

# Phase 10 — Quality Control

**Planned**

- Layout validation
- Asset validation
- PPTX validation
- Rendering verification
- Automatic error detection
- Missing asset detection
- Invalid layout detection
- Text overflow detection
- Element collision detection
- Presentation integrity checks

Future quality-control pipeline:

```
Generated Presentation
        ↓
Layout Validation
        ↓
Asset Validation
        ↓
PPTX Validation
        ↓
Rendering Verification
        ↓
Quality Report
```

---

# Phase 11 — React/Vite UI

**Planned**

- Presentation workspace
- Prompt interface
- Slide preview
- Generation progress
- Asset management
- Presentation outline
- Slide navigation
- Generation status
- Error display
- Export controls

---

# Phase 12 — AI Slide Editor

**Planned**

- AI-powered slide editing
- Regenerate slide
- Rewrite content
- Change layout
- Replace image
- Modify theme
- Add/remove elements
- Modify slide structure
- AI-assisted visual editing

Example:

```
"Make this slide more visual"
        ↓
AI analyzes slide
        ↓
Updated slide representation
        ↓
Layout engine
        ↓
Asset engine
        ↓
Updated PPTX
```

---

# Phase 13 — Themes & Templates

**Planned**

- Presentation themes
- Typography systems
- Color systems
- Slide templates
- Custom branding
- Theme presets
- Consistent visual styles
- Corporate presentation templates

---

# Phase 14 — Research & Web Grounding

**Planned**

- Web research
- Source collection
- Fact grounding
- Citations
- Research-assisted presentations
- Source-aware content generation
- Evidence-backed presentation content

---

# Phase 15 — Project Management & Export

**Planned**

- Presentation projects
- Save/load projects
- Export PPTX
- Asset management
- Presentation history
- Project sessions
- Generated asset reuse
- Presentation versioning

---

# Local AI Models

## Language Model

- Qwen3-4B
- GGUF
- llama.cpp
- Local inference
- CUDA GPU offloading

Used for:

- Content generation
- Presentation planning
- Slide structure
- Asset requirements
- Visual specifications

---

## Image Model

- FLUX.1-schnell
- Q2_K GGUF
- Diffusers
- CUDA
- GPU/RAM/SSD offloading

Used for:

- Presentation imagery
- Scientific illustrations
- Conceptual visualizations
- Technology visuals
- Background imagery
- Other image-like assets

---

# Asset Generation Strategy

The system does not use image generation for every visual element.

Different asset types use different generation strategies.

```
Asset Type
    │
    ├── image
    │      ↓
    │     FLUX
    │
    ├── illustration
    │      ↓
    │     FLUX
    │
    ├── photo
    │      ↓
    │     FLUX
    │
    ├── icon
    │      ↓
    │     SVG / Vector
    │
    ├── diagram
    │      ↓
    │     SVG / Native PPT
    │
    └── chart
           ↓
        Deterministic
        Chart Generator
```

This keeps important presentation elements editable and avoids using generative image models for data-sensitive content.

---

# Structured Visual Specifications

The Presentation Planner produces structured visual specifications for assets.

Example:

```
{
  "type": "image",
  "description": "Quantum system components",
  "visual_spec": {
    "purpose": "Show quantum system components",
    "subject": "Superconducting quantum processor in cryogenic environment",
    "composition": "Center the processor in the lower middle with cryogenic components surrounding it",
    "style": "scientific illustration",
    "color_palette": [
      "deep blue",
      "white"
    ],
    "must_show": [
      "Quantum processor",
      "Cryogenic components",
      "Control wiring"
    ],
    "must_avoid": [
      "text labels",
      "decorative elements"
    ],
    "text_policy": "no text"
  }
}
```

This structured representation allows the asset-generation layer to remain independent from the language model.

---

# Asset Manifest

Each generated asset stores generation metadata.

Example:

```
{
  "id": "asset-005",
  "type": "image",
  "description": "Quantum system components",
  "visual_spec": {},
  "prompt": "...",
  "source": "flux",
  "model": "FLUX.1-schnell",
  "runtime": "diffusers",
  "seed": 42,
  "width": 1024,
  "height": 1024,
  "steps": 4,
  "status": "completed",
  "path": "assets/asset-005.png"
}
```

This allows generation to be:

- Reproducible
- Resumable
- Debuggable
- Inspectable
- Cacheable

---

# Hardware Target

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

The architecture is designed to avoid requiring large GPU VRAM by using hardware-aware model offloading.

---

# Design Principles

- Local-first
- Editable output
- Modular architecture
- Deterministic presentation rendering
- Structured intermediate representation
- Structured visual specifications
- Hardware-aware AI inference
- Efficient memory utilization
- Minimal cloud dependency
- Resumable generation
- Separation of planning and rendering
- Separation of asset planning and asset generation
- Specialized generators for different asset types

---

# Repository Structure

```
Local-agents/
│
├── backend/
│   │
│   ├── ai/
│   │   └── model_manager.py
│   │
│   ├── planner/
│   │   └── presentation_planner.py
│   │
│   ├── ir/
│   │   └── presentation_ir.py
│   │
│   ├── layout/
│   │   ├── geometry.py
│   │   └── layout_engine.py
│   │
│   ├── assets/
│   │   ├── generators/
│   │   ├── asset_router.py
│   │   ├── asset_prompt_builder.py
│   │   └── image_model_manager.py
│   │
│   ├── generation/
│   │   ├── asset_generation_manager.py
│   │   ├── generation_session.py
│   │   └── manifest.py
│   │
│   └── renderers/
│       └── pptx_renderer.py
│
├── config/
│   └── models.json
│
├── generated/
│   └── presentations/
│
├── backend/tests/
│
└── Readme.md
```

---

# Generation Lifecycle

The system manages model memory explicitly.

## Qwen

```
Load Qwen
    ↓
Generate presentation plan
    ↓
Unload Qwen
    ↓
Release GPU/CPU resources
```

## FLUX

```
Load FLUX
    ↓
Generate image 1
    ↓
Generate image 2
    ↓
Generate image 3
    ↓
Unload FLUX
    ↓
Delete temporary SSD offload cache
```

This prevents both large models from unnecessarily occupying GPU memory at the same time.

---

# Development Philosophy

The project is built around a separation between:

```
AI reasoning
      ↓
structured representation
      ↓
deterministic execution
      ↓
editable output
```

The language model should decide **what the presentation should contain**.

The deterministic systems should decide **how the presentation is physically constructed**.

The asset generators should decide **how individual visual assets are produced**.

The PPTX renderer should produce the final editable PowerPoint.

---

# Long-Term Vision

The final system is intended to become a fully local AI presentation platform capable of:

```
Natural Language
      ↓
AI Research
      ↓
AI Planning
      ↓
Content Generation
      ↓
Visual Planning
      ↓
Local Asset Generation
      ↓
Automatic Layout
      ↓
Editable PowerPoint
      ↓
AI Slide Editing
```

with minimal dependence on external cloud AI services.

```

One correction I made from your old README: **Phase 7 is now “In Progress,” not simply “Next”**, because the production `AssetGenerationManager`, manifest-driven generation, visual-spec propagation, and `AssetPromptBuilder` are already implemented. The current blocker is the FLUX prompt-length optimization and the `none` asset handling.
```