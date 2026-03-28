# Geotag 🛰️

> **From satellite imagery to labeled datasets in minutes—not hours.**

Geotag is a standalone, high-performance geospatial labeling application designed for machine learning teams who need quality training data fast. Stream Sentinel-2 imagery directly from the cloud, automatically segment it into intelligent superpixels, and rapidly classify land cover—all through an intuitive visual interface.

---

## ✨ The Experience

### Why Geotag Exists

Creating labeled geospatial datasets traditionally requires:
- Downloading massive GeoTIFF files (hours)
- Manual polygon drawing in desktop GIS software (days)
- Export management and format conversion (frustration)

**Geotag compresses this into minutes.** Draw your area. Stream the imagery. Click to segment. Label with hotkeys. Export to GeoPackage. Done.

---

## 🚀 The Journey: From Zero to Labels

### Step 1: Launch
**Time: 30 seconds**

Open your terminal, activate the environment, start the server:

```bash
conda activate geotag
python -m uvicorn geotag_app:app --reload --host 0.0.0.0 --port 8000
```

Navigate to `http://localhost:8000`. The dark-themed interface loads instantly—no build step, no configuration files.

**You'll see:**
- A full-screen interactive map (Leaflet-powered)
- A sleek sidebar with the Geotag brand
- Progress indicators showing 3 steps: **Stream → Segment → Label**

---

### Step 2: Define Your Area
**Time: 10 seconds**

Click the **rectangle tool** (top-left of the map). Draw a bounding box anywhere on Earth. This is your Area of Interest (AOI).

**What happens:**
- The GeoJSON of your selection appears in the sidebar
- Coordinates are captured in WGS84 (EPSG:4326)
- You're automatically ready to stream

---

### Step 3: Stream Imagery
**Time: 5-30 seconds**

Click the **"Stream Imagery"** button. 

**Behind the scenes:**
1. Geotag queries the [Earth Search STAC API](https://earth-search.aws.element84.com/v1) for Sentinel-2 imagery
2. It finds the clearest, most recent mosaic for your AOI (filtering by cloud cover)
3. Streams 10 spectral bands directly from AWS S3 into memory—no local disk usage
4. Warps the imagery to Web Mercator for display
5. Generates an RGB preview with intelligent contrast stretching

**You'll see:**
- A loading indicator with progress updates
- High-resolution satellite imagery overlays your AOI
- Metadata panel showing capture date, cloud cover %, tile count, and area size

---

### Step 4: Generate Polygons
**Time: 10-60 seconds**

Click **"Generate Polygons"**. This is where the computer vision magic happens.

**Behind the scenes (the segmentation pipeline):**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐
│ Spectral Bands  │──▶│ Felzenszwalb     │───▶│ K-Means     │
│ (Red, NIR,      │    │ Graph-Based      │    │ Clustering  │
│  NDVI, etc.)    │    │ Segmentation     │    │ (n=grn)     │
└─────────────────┘    └──────────────────┘    └─────────────┘
                                                          │
┌─────────────────┐    ┌──────────────────┐               │
│ GeoPackage      │◀──│ Vectorization    │◀─────────────┘
│ (.gpkg)         │    │ (Raster→Polygon) │
└─────────────────┘    └──────────────────┘
```

1. **Felzenszwalb Segmentation**: Groups spectrally-similar pixels into superpixels using a graph-based approach weighted for vegetation signals
2. **Feature Extraction**: Extracts mean + variance signatures from 7 spectral bands per segment
3. **K-Means Clustering**: Groups segments into spectral classes (default: 10 groups)
4. **Vectorization**: Converts raster clusters to GeoJSON polygons with unique IDs

**You'll see:**
- Hundreds to thousands of polygons overlay the imagery
- Each polygon is clickable
- Color-coded by spectral class (for visual distinction)

---

### Step 5: Label with Hotkeys
**Time: 2-10 minutes**

This is the human-in-the-loop step. Click any polygon, then press a number key:

| Key | Class | Color |
|-----|-------|-------|
| `1` | Tree | 🟢 Green |
| `2` | Grassland | 🟩 Light Green |
| `3` | Urban | 🔴 Red |
| `4` | Water | 🔵 Blue |
| `5` | Agriculture | 🟡 Yellow |
| `6` | Barren | 🟤 Brown |
| `0` | Unlabel | ⬜ Gray |

**UX Details:**
- Selected polygons glow with a cyan border
- Assigned labels appear as filled polygons with class-consistent colors
- The status bar shows counts: **"12 labeled / 347 total"**
- **Shift+Click** to multi-select regions
- **Click+Drag** lasso selection for rapid batch labeling

**What happens:**
- Each assignment updates the internal GeoDataFrame
- `class_id` and `label` fields are populated
- Real-time validation prevents empty exports

---

### Step 6: Export
**Time: 1 second**

Click **"Export Labels"**. 

**Output:**
- File: `{project_name}_labels_{YYYYMMDD_HHMM}.gpkg`
- Location: `./exports/{project_name}/labels/`
- Format: OGC GeoPackage (industry standard)
- CRS: Native UTM (preserves area accuracy)
- Contents: Only labeled polygons with `polygon_id`, `class_id`, `label`, `area_sqm`, and geometry

**Use it:**
- Drag into QGIS for visualization
- Load into Python with `geopandas.read_file()`
- Import to ML frameworks (PyTorch Geometric, TensorFlow)
- Feed into training pipelines

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        USER BROWSER                         │
│  ┌─────────────────┐         ┌─────────────────────────────┐│
│  │  React +        │◀──────▶│  Leaflet Map                ││
│  │  Babel (CDN)    │         │  • Draw AOI                 ││
│  │                 │         │  • Display RGB overlay      ││
│  │  • Hotkey       │         │  • Render GeoJSON polygons  ││
│  │    handling     │         │  • Selection interactions   ││
│  └─────────────────┘         └─────────────────────────────┘│
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP/WebSocket
┌──────────────────────────▼──────────────────────────────────┐
│                    FASTAPI BACKEND (Python)                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ /api/stream     │  │ /api/segment    │  │ /api/save    │ │
│  │                 │  │                 │  │              │ │
│  │ • STAC query    │  │ • Felzenszwalb  │  │ • Merge      │ │
│  │ • S3 streaming  │  │ • K-Means       │  │   labels     │ │
│  │ • RGB warp      │  │ • Vectorize     │  │ • GeoPackage │ │
│  │ • Base64 encode │  │ • GeoJSON out   │  │   export     │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           STATE MANAGEMENT (In-Memory)              │    │
│  │  • Current imagery bands (10 Sentinel-2 channels)   │    │
│  │  • Native CRS and geotransform                      │    │
│  │  • Active GeoDataFrame with polygon metadata        │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              EXTERNAL SERVICES (Cloud-Native)               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Earth Search    │  │ AWS S3          │  │ (Your Disk)  │ │
│  │ STAC API        │  │ Sentinel-2      │  │              │ │
│  │                 │  │ COGs            │  │ .gpkg export │ │
│  │ • Item search   │  │ • Stream tiles  │  │              │ │
│  │ • Metadata      │  │ • Cloud-optimized│ │              │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### The Segmentation Engine

The core innovation is the **vegetation-weighted Felzenszwalb → K-Means pipeline**:

1. **Input Stack**: NDVI (×2 weight), RedEdge1, NIR, Blue—optimized for tree/building contrast
2. **Felzenszwalb**: Graph-based segmentation that respects natural boundaries
3. **Parallel Tiling**: Large AOIs (>1.5M px) auto-split into 4 tiles with 1M label offsets
4. **Feature Signature**: 14-dim vector per segment (mean + std of 7 bands)
5. **Spectral Clustering**: K-Means groups segments by vegetation health + texture
6. **Sieve + Vectorize**: Removes noise, raster → polygon with rasterio.features

**Result**: 500-5,000 polygons per sq km, depending on granularity setting (1-50).

---

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- Conda (Miniforge or Anaconda)
- 4GB+ RAM (8GB+ recommended for large AOIs)
- Internet connection (for STAC/S3 streaming)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/mo-labs/geotag.git
cd geotag

# Create conda environment with geospatial stack
conda env create -f environment.yml
conda activate geotag

# Install geotag in editable mode
pip install -e .

# Start the server
python -m uvicorn geotag_app:app --reload --host 0.0.0.0 --port 8000
```

**Access:** Open `http://localhost:8000` in your browser.

---

## 📋 API Reference

### `POST /api/stream`

Stream Sentinel-2 imagery for an AOI.

**Request Body:**
```json
{
  "aoi_geojson": {...},          // GeoJSON FeatureCollection
  "start_date": "2024-06-01",     // ISO date
  "end_date": "2024-06-30",       // ISO date
  "max_cloud_cover": 20,          // 0-100%
  "granularity": 10               // K-Means clusters (affects segment colors)
}
```

**Response:**
```json
{
  "rgb_image": "data:image/png;base64,...",
  "rgb_bounds": [[lat, lon], [lat, lon]],
  "metadata": {
    "date": "2024-06-15T10:30:00Z",
    "cloud_cover": 12.3,
    "tiles": 2,
    "area_sqkm": 50.0
  },
  "duration_sec": 15.2
}
```

---

### `POST /api/segment`

Generate vector polygons from streamed imagery.

**Request Body:** Same as `/api/stream` (reuses current imagery if available).

**Response:**
```json
{
  "geojson": {...},        // GeoJSON FeatureCollection with polygons
  "duration_sec": 42.5
}
```

**Polygon Properties:**
- `polygon_id`: Unique integer ID
- `class_id`: -1 (unlabeled) or assigned class
- `label`: "unlabeled" or user-assigned string
- `area_sqm`: Geometry area in square meters

---

### `POST /api/save`

Export labeled polygons to GeoPackage.

**Request Body:**
```json
{
  "polygon_labels": {
    "0": {"class_id": 1, "label": "Tree"},
    "1": {"class_id": 3, "label": "Urban"}
  },
  "project_name": "riyadh_trees"
}
```

**Response:**
```json
{
  "status": "success",
  "path": "/home/user/geotag/exports/riyadh_trees/labels/riyadh_trees_labels_20250127_1430.gpkg",
  "assigned_polygons": 2
}
```

---

## 🎛️ Configuration

### Granularity Setting

Controls segmentation detail (UI slider: 1-50):

| Value | Use Case | Polygons/sqkm |
|-------|----------|---------------|
| 5 | Large homogeneous areas | ~200 |
| 10 | Balanced (default) | ~500 |
| 25 | Detailed land cover | ~2,000 |
| 50 | Fine-grained analysis | ~5,000 |

Higher values = more polygons = longer processing time.

---

## 🔧 Troubleshooting

### "No imagery found for AOI"
- Increase `max_cloud_cover` (try 40-60%)
- Expand date range (try 3-month window)
- Verify AOI is within Sentinel-2 coverage (globally complete since 2017)

### Slow segmentation
- Enable **"Use Multi-threading"** for large AOIs
- Enable **"Use Downsampling"** for very large AOIs (>100 sq km)
- Reduce granularity for fewer polygons

### Memory errors
- Reduce AOI size (process in tiles)
- Enable downsampling (2x or 4x)
- Close other applications

### Export is empty
- You must label at least one polygon before export
- Check that polygons show "assigned" status in UI

---

## 🧪 Development

### Project Structure

```
geotag/
├── geotag_app.py          # FastAPI backend (453 lines)
├── index.html             # React frontend (1082 lines, self-contained)
├── pyproject.toml         # Package configuration
├── environment.yml        # Conda dependencies
├── exports/               # Output directory (created on first export)
└── README.md              # This file
```

### Backend (`geotag_app.py`)

**Key Dependencies:**
- `fastapi` + `uvicorn`: HTTP server
- `pystac_client`: STAC API queries
- `rioxarray`: Cloud-optimized GeoTIFF streaming
- `rasterio`: Reprojection and warping
- `scikit-image`: Felzenszwalb segmentation
- `scikit-learn`: K-Means clustering
- `geopandas`: Vector data management

**State Management:**
The backend uses simple in-memory state (global `STATE` dict). Not suitable for multi-user deployments without modification.

### Frontend (`index.html`)

A single-file React application loaded via CDN:
- **React 18**: Component framework
- **Leaflet 1.9**: Interactive mapping
- **Leaflet.draw**: AOI drawing tools
- **Babel**: JSX transformation in browser
- **No build step**: Open `index.html` directly or serve via FastAPI

**Key Components:**
- `MapView`: Leaflet instance with layer management
- `Sidebar`: Progress tracking and metadata display
- `LabelPanel`: Class selection and polygon info
- `ExportModal`: Save configuration UI

---

## 📦 Output Format

### GeoPackage Schema

| Column | Type | Description |
|--------|------|-------------|
| `polygon_id` | INTEGER | Unique segment identifier |
| `class_id` | INTEGER | Assigned class (1-6, or -1 for unlabeled) |
| `label` | TEXT | Human-readable class name |
| `area_sqm` | REAL | Area in square meters (UTM projection) |
| `geometry` | POLYGON | WKB geometry in native CRS |

**CRS:** UTM zone of AOI (preserves accurate area measurements)

---

## 🗺️ Roadmap

- [ ] Multi-temporal labeling (compare seasons)
- [ ] Custom class definitions (JSON config)
- [ ] Session persistence (SQLite backend)
- [ ] Multi-user support (authentication)
- [ ] Cloud deployment (Docker + Fly.io template)
- [ ] QGIS plugin for direct import
- [ ] Active learning suggestions (label uncertainty highlighting)

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Issues and PRs welcome at [github.com/mo-labs/geotag](https://github.com/mo-labs/geotag).

**Development Setup:**

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linting
black geotag_app.py
ruff geotag_app.py

# Run tests (when available)
pytest
```

---

<p align="center">
  <strong>Built with ❤️ by Mo Labs</strong><br>
  <em>Democratizing geospatial ML, one label at a time.</em>
</p>
