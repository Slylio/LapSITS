# LapSITS - Pattern Spectra Viewer

## Project Description

LapSITS is an application for analysis and visualization of temporal image sequences (SITS - Satellite Image Time Series) using Pattern Spectra based on component trees.

### Main Features

1. **SITS Visualization**: Display of image sequences with temporal navigation
2. **Pattern Spectra Calculation**: Automatic generation of global PS based on Area vs Stability
3. **Bidirectional Selection**:
   - **Image → PS**: Polygon selection in the image to highlight corresponding bins in the PS
   - **PS → Image**: Bin selection in the PS to visualize corresponding nodes in the SITS

## Project Structure

```
LapSITS/
├── main.py                 # Application entry point
├── core/
│   └── pattern_spectra.py  # Pattern Spectra calculations
├── gui/
│   ├── main_window.py      # Main window
│   ├── image_canvas.py     # Canvas for SITS images
│   └── ps_canvas.py        # Interactive canvas for Pattern Spectra
└── data/
    └── sits_example/       # Time sequence images
```

## User Guide

### Main Interface

The interface is divided into two parts:
- **Left**: SITS image visualization with controls
- **Right**: Interactive Pattern Spectra

### Image Selection (Image → PS)

1. **Create a polygon**:
   - Left click to add points
   - Right click to close the polygon
   - Corresponding bins automatically light up in the PS

2. **Temporal navigation**:
   - Use the slider to change timestep
   - The polygon remains active and corresponding bins are updated

### Pattern Spectra Selection (PS → Image)

1. **Click Mode**:
   - Activate "Click" in the controls
   - Click on PS bins to select/deselect them
   - Hold left mouse button and drag to select multiple bins
   - Corresponding nodes light up in red in the image

2. **Polygon Mode**:
   - Activate "Polygon" in the controls
   - Left click to add points to the polygon
   - Polygon closes automatically when you have enough points
   - All contained bins are selected

3. **Temporal navigation**:
   - Change timestep with the slider
   - Areas affected by selected nodes are visualized for each time

### Keyboard Shortcuts

**In the image**:
- `Escape`: Cancel current polygon selection
- `Delete`: Clear current polygon

**In the Pattern Spectra**:
- `c`: Click mode
- `p`: Polygon mode
- `Escape`: Cancel current selection
- `Delete`: Clear all selection

### Control Buttons

- **Recalculate PS**: Recalculates the Pattern Spectra (useful after data changes)
- **Clear Selection**: Clears all selections (image and PS)
- **PS Mode**: Choose between Click and Polygon selection modes for the Pattern Spectra

## Algorithms Used

### Pattern Spectra
- **Component Tree**: 3D Tree of Shapes to capture spatio-temporal structure
- **Attributes**:
  - Area: Component area (logarithmic scale)
  - Stability: Temporal stability of components (linear scale)

### Bidirectional Selection
- **Image → PS**: Uses 3D masks and tree propagation
- **PS → Image**: Goes from bins to nodes then to pixels per timestep

### Performance Optimizations
- **Caching**: Node masks are cached to avoid recalculation
- **Optimized rendering**: Base plots are cached when possible
- **Efficient overlay creation**: Direct RGBA image creation for overlays

## Technologies

- **Interface**: PyQt5
- **Calculations**: Higra (component trees), NumPy
- **Visualization**: Matplotlib integrated in PyQt5
- **Geometry**: Shapely for polygon handling

## Development

The code is organized in a modular way:
- `core/`: Business logic and algorithms
- `gui/`: User interface and interactions
- Clear separation between calculations and display
- PyQt5 signals for communication between components

## Performance Notes

The application uses several optimization techniques:
1. **Cached mask calculation**: Node masks are computed once and reused
2. **Optimized polygon selection**: Uses efficient algorithms for bin-in-polygon tests
3. **Smart rendering**: Only redraws when necessary
4. **Memory-efficient overlays**: Uses optimized image creation for large datasets
