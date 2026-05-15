# Bentron/Pemtron AOI PTT Viewer

Utilities for inspecting Bentron/Pemtron AOI `.ptt` 3D files, companion `.pot` data, and required `.jpg` textures.

The main tool is `view_bentron_aoi_3d_gl.py`, an OpenGL viewer for interactive textured 3D rendering. The current default height mode is `plane0_repair`: it preserves the observed best `plane0` geometry, repairs small `plane0` invalid holes as low surface, and keeps the `K` culling path from cutting repaired PCB holes into black gaps.

## Install

```powershell
python -m pip install -r requirements.txt
```

Dependencies:

- `numpy`
- `Pillow`
- `pyglet`

## Run

Open a file picker:

```powershell
python .\view_bentron_aoi_3d_gl.py
```

Open one file:

```powershell
python .\view_bentron_aoi_3d_gl.py .\samples\1@206.ptt
```

Each `.ptt` must have a companion texture next to it:

- `sample.jpg`
- or `sample_AC.jpg`

The `.ptt` files observed so far contain only a header plus 3 height planes, with no embedded texture payload. If neither JPG exists, the viewer reports a missing texture error instead of opening a partial render.

Open a folder of samples:

```powershell
python .\view_bentron_aoi_3d_gl.py .\samples
```

Useful options:

```powershell
python .\view_bentron_aoi_3d_gl.py .\samples\1@836.ptt --grid 360 --height-mode plane0_repair
```

Height modes:

- `plane0`: raw primary height plane.
- `plane0_repair`: default; `plane0` with invalid holes repaired as low surface.
- `plane1`, `plane2`: raw secondary planes for inspection.
- `mean`, `weighted`: experimental plane averaging.
- `fill_min12`, `fill_qmap12`: experimental fill modes using planes 1/2.

## Controls

- Drag: rotate freely
- Wheel: zoom
- `O`: open another `.ptt`
- `N` / `P`: next/previous sample in the folder
- `+` / `-`: height display scale
- `W/A/S/D` or arrow keys: texture offset
- `U`: reset texture offset
- `X`: flip left/right
- `T`: switch texture
- `V`: debug texture view. The `height` view shows the current mesh height actually used for 3D rendering, after repair/capping/resizing.
- `G`: height/fusion recipe
- `K`: cull invalid faces after hole repair
- `H`: specular strength
- `B`: bump detail
- `M`: mesh smoothing passes
- `L`: light direction
- `C`: save alignment
- `E`: export depth maps
- `Q`: pick diagnostics for a clicked point
- `F1`: show/hide help
- `Esc`: close

`Q` pick diagnostics export JSON, screenshot, plane patches, POT patch, and texture patch to `pick_diagnostics/`. Use this when comparing abnormal render points against raw `plane0/1/2` and `.pot` values.

## Format Notes

Observed `.ptt` layout:

- 76-byte header
- first two `uint32` fields are height and width
- next two `float32` fields are pixel pitch
- payload is 3 little-endian `uint16` planes shaped `(3, height, width)`
- values near `65535` behave as invalid/sentinel heights

Observed `.pot` layout:

- 20-byte header
- 5 planar `uint8` channels

The official `PEM3DControl.ocx` can be hosted for comparison. The probe in `official_ocx_probe/` uses a minimal WinForms ActiveX host and can call methods such as `LoadFile`, `GetHeightBuffer`, `GetRealBuffer`, and `GetHeightMinMax`. The OCX is not suitable for browser rendering, but it is useful as a reference renderer.

## OCX Probe

The probe expects the vendor files under `Pemtron-PROJECT/`, including `PEM3DControl.ocx`, `AxInterop.PEM3DControlLib.dll`, and `Interop.PEM3DControlLib.dll`.

```powershell
dotnet build .\official_ocx_probe\OfficialOcxProbe.csproj -c Release
.\official_ocx_probe\bin\Release\net8.0-windows\win-x64\OfficialOcxProbe.exe .\samples\1@206.ptt .\official_ocx_probe\out_206
```

If COM activation fails, register the OCX/type library for the current user before running the probe.

## Tests

```powershell
python -m unittest .\test_viewer_core.py
python -m py_compile .\view_bentron_aoi_3d_gl.py .\test_viewer_core.py
```

The tests use local samples when present. Sample files are intentionally ignored.

## Packaging

Build the standalone viewer with PyInstaller:

```powershell
pyinstaller .\Pemtron3DViewer.spec --clean --noconfirm
```

The executable is written to:

```text
dist\Pemtron3DViewer.exe
```

The packaged viewer includes a Windows native file picker fallback, so it does not require `tkinter`.

## Repository Scope

Large vendor binaries, sample data, generated renders, diagnostics, and packaged outputs are ignored. Keep source code, tests, packaging metadata, and small project files in Git.
