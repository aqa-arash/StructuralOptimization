# Density XML Initialization Strategy

## Overview

The "from_density" initialization strategy allows you to initialize feature-based topology optimization directly from an existing density field stored in a `*.density.xml` file. This is useful for:

- Warm-starting optimizations from previous results
- Extracting existing design patterns and refining them
- Reading grid resolution and boundary parameters from a reference density file

## Architecture

The implementation consists of three components:

### 1. `density_xml_utils.py` - Utility Functions

This module provides core functions for reading and writing density XML files:

- **`read_density_xml(xml_path)`**: Parses a density XML file and returns the density grid and metadata (grid size, internal_transition, external_transition, bounds)
- **`extract_shapes_from_density(density_grid, threshold, num_shapes, bounds)`**: Identifies connected components in the density field and extracts their positions, orientations, and sizes using PCA
- **`write_density_xml(density_grid, output_path, internal_transition, external_transition, ...)`**: Writes a density field back to XML format with proper transition parameters

### 2. `optimize.py` - Configuration Update Function

- **`update_config_from_density_xml(config, density_xml_path, threshold)`**: Reads a density XML file, extracts shapes, and updates the optimization config with:
  - Grid resolution (nx, ny)
  - Internal transition and external transition parameters
  - Number of features detected
  - Extracted shape positions and orientations

### 3. `initialize_features()` - New Strategy Support

The `initialize_features()` function now accepts a new strategy:

```python
initialize_features(
    strategy="from_density",
    num_features=...,  # Will use detected shapes count
    start_radius=...,
    shapes_data=extracted_shapes,  # List of detected shapes
    bounds=...,
    ...
)
```

## Usage

### Step 1: Create Configuration with from_density Strategy

Create a `config.json` file with the `from_density` strategy:

```json
{
  "start_strategy": "from_density",
  "density_xml_path": "path/to/your/reference.density.xml",
  "density_threshold": 0.5,
  "...": "other config options"
}
```

### Step 2: Update Config Before Optimization (Python)

Before running optimization, call `update_config_from_density_xml()` to populate the config:

```python
from optimize import update_config_from_density_xml, load_config

# Load config
config = load_config("config.json")

# Update config from density file
if config.get("start_strategy") == "from_density":
    config = update_config_from_density_xml(
        config,
        config["density_xml_path"],
        threshold=config.get("density_threshold", 0.5)
    )

# Now run optimization with updated config
# The grid resolution, feature count, etc. are automatically set
```

### Step 3: Run Optimization

The optimization will initialize features directly from the density XML positions:

```python
from optimize import run_configured_optimization

run_configured_optimization(
    density_path=None,  # Or path to target density if needed
    config=config,
    output_dir="results"
)
```

## How It Works

1. **Read Density XML**: The utility reads the grid resolution, transition/extension parameters, and optimization bounds from the XML header
2. **Extract Shapes**: Connected components above the density threshold are identified and analyzed
3. **Compute Geometry**: For each shape, the centroid, major axis orientation, and extent are computed using PCA
4. **Initialize Features**: Each detected shape is converted into a feature segment (P, Q, r) aligned with its principal axis
5. **Clip to Bounds**: Feature endpoints are clipped to the specified optimization domain bounds

## Example Configuration

```json
{
  "start_strategy": "from_density",
  "density_xml_path": "results/previous_run.density.xml",
  "density_threshold": 0.5,
  "grid_resolution": [100, 100],
  "internal_transition": 0.1,
  "external_transition": 0.0,
  "boundary": "bezier",
  "start_radius": 0.05,
  "stages": [
    {
      "description": "Fine-tune existing design",
      "reward_only": false,
      "n_iter": 300,
      "tolerance": 1e-8
    }
  ]
}
```

## Notes

- The `density_threshold` parameter controls which density regions are considered as features (default: 0.5)
- Grid resolution is automatically updated from the XML file
- The strategy works best with reasonable density contrast (avoid very gradual transitions)
- For better results, the reference density file should have been created with the same boundary method ("bezier" recommended)
- All feature endpoints are clipped to the optimization domain bounds to ensure validity
