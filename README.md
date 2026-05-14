# Bentron/Pemtron AOI PTT Viewer

Utilities for exploring Bentron/Pemtron AOI 3D sample files such as `.ptt`, `.pot`, and companion `.jpg` textures.

## Current Tools

- `view_bentron_aoi_3d_gl.py`: OpenGL textured 3D viewer for `.ptt` samples.
- `export_ptt_depth.py`: Export the raw `plane0` depth map from a `.ptt` file.
- `render_bentron_aoi.py`: Export and inspect raw `.ptt` / `.pot` planes.
- `render_bentron_aoi_3d.py`: Static/HTML 3D render prototype.
- `view_bentron_aoi_3d.py`: Older software-rendered viewer kept for reference.

## Install

```powershell
python -m pip install -r requirements.txt
```

The Codex bundled Python used during development was:

```powershell
C:\Users\sugar\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
```

## Run The OpenGL Viewer

Open a file picker:

```powershell
python .\view_bentron_aoi_3d_gl.py
```

Open a specific sample:

```powershell
python .\view_bentron_aoi_3d_gl.py .\samples\1@206.ptt
```

Useful controls:

- Drag: rotate freely
- Wheel: zoom
- `O`: open another `.ptt`
- `N` / `P`: next/previous sample in the same folder
- `+` / `-`: height display scale
- `W/A/S/D` or arrow keys: adjust texture offset
- `U`: reset texture offset
- `X`: flip left/right
- `T`: switch texture
- `H`: specular
- `B`: bump detail
- `M`: mesh smoothing passes
- `V`: cycle debug texture views: original texture, height color map, board mask overlay, POT RGB composite
- `L`: light direction
- `C`: save current alignment
- `E`: export current depth maps
- `F1`: show/hide help
- `Esc`: close

## Format Notes

Observed `.ptt` layout:

- 76-byte header
- 3 little-endian `uint16` planes
- dimensions from the first two `uint32` header fields
- pixel pitch from the following two `float32` fields

Observed `.pot` layout:

- 20-byte header
- 5 planar `uint8` channels
- header contains width, height, and pitch values

The current OpenGL viewer uses `plane0` as the main height source, a companion `.jpg` as the texture, and derives a board baseline from the aligned image. The official Pemtron OCX exposes methods such as `SetFilterMode`, `SetCutOffLevel`, and `GetHeightBuffer`; those are likely relevant for matching the official renderer more closely.

## Testing

The tests use local samples when present, but sample files are intentionally not committed.

```powershell
python -m unittest .\test_viewer_core.py
```

Current checks cover:

- `.ptt` layout parsing
- `.pot` layout parsing
- `flipX` preserving the height distribution
- debug texture generation

## Repository Scope

Large vendor binaries, sample data, generated renders, and debug images are intentionally ignored. Keep only source code and small project metadata in Git.
