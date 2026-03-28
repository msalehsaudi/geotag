"""
FastAPI Backend for Interactive Geospatial Labeling.
Handles STAC querying, S3 streaming, K-Means segmentation, and GeoJSON serving.
"""

import logging
import os
import time
from time import perf_counter
from typing import List, Optional
from pathlib import Path
import io
import base64
from PIL import Image
import numpy as np
import pandas as pd
import geopandas as gpd
import yaml
import rasterio.transform
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from rasterio.features import shapes, sieve
from shapely.geometry import box, mapping, shape
from skimage.segmentation import slic, felzenszwalb
from skimage import graph
from scipy.ndimage import label, generate_binary_structure
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pystac_client import Client
from sklearn.cluster import KMeans
import rioxarray
import rioxarray.merge
from joblib import Parallel, delayed

# GDAL Env vars to prevent Earth Search AWS hanging issues
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["AWS_REQUEST_PAYER"] = "requester"
os.environ["AWS_REGION"] = "us-west-2"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Quiet boto3/rasterio logs
logging.getLogger("rasterio").setLevel(logging.WARNING)

# --- STAC / S3 Configuration ---
EARTHSEARCH_URL = "https://earth-search.aws.element84.com/v1"
S2_COLLECTION = "sentinel-2-c1-l2a"

# Mapping to all useful Sentinel-2 L2A bands
BAND_MAPPING = {
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "rededge1": "B05",
    "rededge2": "B06",
    "rededge3": "B07",
    "nir": "B08",
    "nir08": "B8A",
    "swir16": "B11",
    "swir22": "B12"
}


def get_s3_href(asset):
    """Extract S3 URL from STAC asset."""
    if "alternate" in asset.extra_fields and "s3" in asset.extra_fields["alternate"]:
        return asset.extra_fields["alternate"]["s3"]["href"]
    href = asset.href
    if href.startswith("https://") and ".s3." in href:
        parts = href.split(".s3.")
        bucket = parts[0].replace("https://", "")
        key = parts[1].split(".amazonaws.com/", 1)[-1]
        return f"s3://{bucket}/{key}"
    return href


def fetch_lazy_mosaic(items, asset_names, aoi_geom_wgs84):
    """
    Given a list of STAC items (usually from the same timestamp) and requested assets,
    open them lazily, clip/reproject each to the AOI, and merge into a single mosaic.
    """
    band_arrays = {}
    
    # Use the first asset of the first item to define the reference CRS (UTM)
    first_asset = asset_names[0]
    href_first = get_s3_href(items[0].assets[first_asset])
    da_ref = rioxarray.open_rasterio(href_first, chunks={"x": 512, "y": 512})
    native_crs = da_ref.rio.crs
    
    aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_geom_wgs84], crs="EPSG:4326")
    aoi_native = aoi_gdf.to_crs(native_crs)
    native_bbox = tuple(aoi_native.total_bounds)
    
    # Pre-clip each item per asset, then merge
    for a_name in asset_names:
        tile_arrays = []
        for item in items:
            href = get_s3_href(item.assets[a_name])
            da = rioxarray.open_rasterio(href, chunks={"x": 512, "y": 512})
            clipped = da.rio.clip_box(*native_bbox)
            tile_arrays.append(clipped)
        
        # Merge all tiles for this band
        merged = rioxarray.merge.merge_arrays(tile_arrays)
        
        # Note: we should still ensure the grid matches the first band's final extent
        if a_name == first_asset:
            da_ref_mosaic = merged
        else:
            merged = merged.rio.reproject_match(da_ref_mosaic)
            
        merged = merged.compute()
        band_arrays[a_name] = merged.values.squeeze().astype(np.float32)

    # Recreate transform for the mosaic
    res_x = float(abs(da_ref_mosaic.x[1] - da_ref_mosaic.x[0]))
    res_y = float(abs(da_ref_mosaic.y[1] - da_ref_mosaic.y[0]))
    west = float(da_ref_mosaic.x.min()) - (res_x / 2.0)
    north = float(da_ref_mosaic.y.max()) + (res_y / 2.0)
    transform = rasterio.transform.from_origin(west, north, res_x, res_y)
            
    # Area in Hectares
    aoi_area_ha = aoi_native.geometry.iloc[0].area / 10000.0
            
    return {
        "native_crs": native_crs,
        "transform": transform,
        "bands": band_arrays,
        "aoi_area_ha": aoi_area_ha
    }


def fetch_lazy_ds(item, asset_names, aoi_geom_wgs84):
    """Legacy wrapper for single item."""
    return fetch_lazy_mosaic([item], asset_names, aoi_geom_wgs84)

# --- RAG Merging Helpers ---
def merge_mean_color(graph, src, dst):
    """Callback for merging two nodes in a Region Adjacency Graph."""
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])

def standard_weight(graph, src, dst, n):
    """Callback to compute weight between merged node and neighbor."""
    return np.linalg.norm(graph.nodes[dst]['mean color'] -
                          graph.nodes[n]['mean color'])

app = FastAPI(title="Treelance Labeling API")

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    for route in app.routes:
        methods = getattr(route, "methods", "N/A")
        logger.info(f"Route: {route.path} - Methods: {methods}")

@app.get("/")
async def get_index():
    return FileResponse("index.html")

# Also serve current dir as static
app.mount("/static", StaticFiles(directory="."), name="static")

# --- Parallel Helpers ---
def _process_tile(tile, scale, sigma, min_size, label_offset):
    """Worker function for Felzenszwalb tiling."""
    # felzenszwalb is imported inside to avoid pickle issues if needed
    from skimage.segmentation import felzenszwalb
    segs = felzenszwalb(tile.astype(np.float64), scale=scale, sigma=sigma, min_size=min_size)
    return (segs + label_offset).astype(np.int32)

class SegmentRequest(BaseModel):
    config_path: str = "configs/test_50sqkm.yaml"
    granularity: int = 10
    use_multi: bool = True
    use_downsampling: bool = False
    aoi_geojson: Optional[dict] = None
    start_date: str = "2024-06-01"
    end_date: str = "2024-06-30"
    max_cloud_cover: int = 20

class SaveRequest(BaseModel):
    polygon_labels: dict # Map of polygon_id (str) to {"class_id": int, "label": str}
    project_name: str = "default_project"
    output_name: Optional[str] = None

# Global state to store the current segment result for the session
# In a production app, this would be in a database or cache
STATE = {
    "gdf": None,
    "native_crs": None,
    "current_data": None
}

@app.post("/api/stream")
async def stream_imagery(request: SegmentRequest):
    """
    STAC search and RGB imagery streaming.
    Returns RGB image and bounds for display.
    """
    t0 = time.perf_counter()
    try:
        if request.aoi_geojson:
            logger.info("Using provided GeoJSON AOI from frontend")
            aoi_df = gpd.GeoDataFrame.from_features(request.aoi_geojson["features"], crs="EPSG:4326")
        else:
            with open(request.config_path, "r") as f:
                config = yaml.safe_load(f)
            input_file = config.get("input", {}).get("aoi")
            aoi_df = gpd.read_file(input_file)
            
        bbox = list(aoi_df.total_bounds)
        aoi_geom_wgs84 = shape(mapping(aoi_df.geometry.iloc[0]))
        
        client = Client.open(EARTHSEARCH_URL)
        search = client.search(
            collections=[S2_COLLECTION],
            bbox=bbox,
            datetime=f"{request.start_date}/{request.end_date}",
            query={"eo:cloud_cover": {"lt": request.max_cloud_cover}},
            sortby=[{"field": "properties.datetime", "direction": "asc"}]
        )
        items = list(search.items())
        if not items:
            raise HTTPException(status_code=404, detail="No imagery found for AOI")
            
        # Group items by date to handle AOIs spanning multiple tiles (mosaic)
        # We group by YYYY-MM-DD to ensure passes crossing hour boundaries are merged.
        items_by_day = {}
        for it in items:
            d_key = it.properties["datetime"][:10]
            if d_key not in items_by_day:
                items_by_day[d_key] = []
            items_by_day[d_key].append(it)
            
        # Select the date with the lowest average cloud cover
        def avg_cc(day_items):
            return sum(it.properties.get("eo:cloud_cover", 100) for it in day_items) / len(day_items)
            
        best_day = min(items_by_day.keys(), key=lambda k: avg_cc(items_by_day[k]))
        mosaic_items = items_by_day[best_day]
        first_time = best_day # Used for metadata reporting
        
        asset_names = list(BAND_MAPPING.keys())
        logger.info(f"Streaming mosaic of {len(mosaic_items)} items for {first_time}")
        
        data = fetch_lazy_mosaic(mosaic_items, asset_names, aoi_geom_wgs84)
        
        # Area in sqkm (Calculate from native geometry after fetch_lazy_mosaic sets the CRS)
        aoi_area_sqkm = round(data["aoi_area_ha"] / 100.0, 4) # ha to sqkm
        
        # Metadata for info panel
        primary_date = mosaic_items[0].properties.get("datetime")
        avg_cloud = np.mean([it.properties.get("eo:cloud_cover", 0) for it in mosaic_items])
        
        # Store data in state for later segmentation
        STATE["current_data"] = data
        STATE["native_crs"] = data["native_crs"]
        
        # Calculate Base64 RGB Image Overlay — warped to EPSG:4326 for alignment
        red = data["bands"]["red"]
        green = data["bands"]["green"]
        blue = data["bands"]["blue"]
        
        height, width = red.shape
        native_crs = data["native_crs"]
        native_transform = data["transform"]
        dst_crs = CRS.from_epsg(4326)
        
        # Compute the target transform and dimensions in WGS84
        bounds_native = rasterio.transform.array_bounds(height, width, native_transform)
        dst_transform, dst_width, dst_height = calculate_default_transform(
            native_crs, dst_crs, width, height,
            left=bounds_native[0], bottom=bounds_native[1],
            right=bounds_native[2], top=bounds_native[3]
        )
        
        # Warp each band to WGS84
        rgb_warped = []
        for band_arr in [red, green, blue]:
            dst_arr = np.zeros((dst_height, dst_width), dtype=np.float32)
            reproject(
                source=band_arr.astype(np.float32),
                destination=dst_arr,
                src_transform=native_transform,
                src_crs=native_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )
            rgb_warped.append(dst_arr)
        
        rgb_stack = np.stack(rgb_warped, axis=-1)
        # Contrast stretch
        p2, p98 = np.nanpercentile(rgb_stack[rgb_stack > 0], (2, 98))
        rgb_stack = np.clip((rgb_stack - p2) / (p98 - p2 + 1e-8), 0, 1)
        rgb_uint8 = (rgb_stack * 255).astype(np.uint8)
        
        img = Image.fromarray(rgb_uint8)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        rgb_base64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        
        # Bounds are now directly in WGS84 from the warped transform
        bounds_wgs84 = rasterio.transform.array_bounds(dst_height, dst_width, dst_transform)
        leaflet_bounds = [[bounds_wgs84[1], bounds_wgs84[0]], [bounds_wgs84[3], bounds_wgs84[2]]]
        
        duration_sec = round(time.perf_counter() - t0, 1)
        return {
            "rgb_image": rgb_base64,
            "rgb_bounds": leaflet_bounds,
            "metadata": {
                "date": primary_date,
                "cloud_cover": round(float(avg_cloud), 1),
                "tiles": len(mosaic_items),
                "area_sqkm": round(aoi_area_sqkm, 2)
            },
            "duration_sec": duration_sec
        }

    except Exception as e:
        logger.exception("Streaming failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/segment")
async def segment_aoi(request: SegmentRequest):
    t0 = time.perf_counter()
    """
    K-Means segmentation on the ALREADY STREAMED imagery.
    """
    if STATE["current_data"] is None:
        raise HTTPException(status_code=400, detail="Must stream imagery first")
    try:
        data = STATE["current_data"]
        asset_names = list(BAND_MAPPING.keys())
        h, w = data["bands"]["red"].shape
        
        # 1. Normalize all bands to 0-1
        norm_bands = {}
        for b in asset_names:
            arr = data["bands"][b]
            norm_bands[b] = (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr) + 1e-8)
        
        t_prep = time.perf_counter() - t0
        logger.info(f"PERF: Band normalization took {t_prep:.2f}s")
            
        # 2. Felzenszwalb Segmentation (Graph-based, vegetation-weighted)
        t_seg_start = perf_counter()
        scale_val = max(5.0, 1000.0 / request.granularity)
        
        # We synthesize an 'Edge Search Stack' prioritized by vegetation signals
        ndvi = (norm_bands["nir"] - norm_bands["red"]) / (norm_bands["nir"] + norm_bands["red"] + 1e-8)
        ndvi_norm = (ndvi + 1) / 2.0
        
        # Stack: NDVI, RedEdge1 (B05), NIR, Red for maximum tree/building contrast
        seg_stack = np.stack([
            ndvi_norm * 2.0,     # Heavily weight NDVI for boundaries
            norm_bands["rededge1"],
            norm_bands["nir"], 
            norm_bands["blue"] # Keep blue for some urban edge shadow detection
        ], axis=-1)
        
        # --- SEGMENTATION ENGINE ---
        if request.use_multi and (h * w) > 1_500_000:
            logger.info(f"Using Parallel Tiling on {h}x{w} grid...")
            # Divide into 2x2 grid (4 tiles)
            mid_h, mid_w = h // 2, w // 2
            tiles = [
                seg_stack[0:mid_h, 0:mid_w],
                seg_stack[0:mid_h, mid_w:w],
                seg_stack[mid_h:h, 0:mid_w],
                seg_stack[mid_h:h, mid_w:w]
            ]
            
            # Offsets to ensure globally unique labels
            # We use 1M as a safe gap between tile label ranges
            offsets = [0, 1_000_000, 2_000_000, 3_000_000]
            
            # Run in parallel
            results = Parallel(n_jobs=-1)(
                delayed(_process_tile)(t, scale_val, 0.0, 1, off) 
                for t, off in zip(tiles, offsets)
            )
            
            # Reassemble
            top = np.hstack([results[0], results[1]])
            bottom = np.hstack([results[2], results[3]])
            segments = np.vstack([top, bottom])
            
        elif request.use_downsampling and (h * w) > 2_000_000:
            ds = 2 if (h * w) < 8_000_000 else 4
            logger.info(f"Using Adaptive Downsampling ({ds}x) on {h}x{w} grid...")
            ss_ds = seg_stack[::ds, ::ds]
            segs_ds = felzenszwalb(ss_ds.astype(np.float64), scale=scale_val, sigma=0.0, min_size=1)
            segments = np.repeat(np.repeat(segs_ds, ds, axis=0), ds, axis=1)[:h, :w]
        
        else:
            logger.info(f"Running Single-Threaded Felzenszwalb on {h}x{w} grid...")
            segments = felzenszwalb(seg_stack.astype(np.float64), scale=scale_val, sigma=0.0, min_size=1)

        t_seg = perf_counter() - t_seg_start
        logger.info(f"PERF: Core Segmentation took {t_seg:.2f}s")
        segments = (segments + 1).astype(np.int32) # Ensure 1-based labels

        # 3. Create Feature Signature Stack (Comprehensive Spectral Profile)
        t_feat_start = perf_counter()
        # Include RedEdge and SWIR for superior urban/forest separation
        feat_list = [
            norm_bands[b] for b in ["red", "green", "blue", "nir", "rededge1", "swir16"]
        ]
        feat_list.append(ndvi)
        stack = np.stack(feat_list, axis=-1)
        
        # 4. Extract segment-level signatures (Means + Texture Variance)
        unique_segs = np.unique(segments)
        region_features = []
        for sid in unique_segs:
            mask = (segments == sid)
            pixels = stack[mask]
            
            # Mean color/NDVI + std (texture)
            mean = pixels.mean(axis=0)
            std = pixels.std(axis=0)
            
            # BIAS: Multiply vegetation signals to dominate the clustering
            # index 3=NIR, 6=NDVI, 4=RedEdge
            mean[6] *= 8.0  # NDVI is king for trees
            mean[3] *= 4.0  # NIR for biomass
            mean[4] *= 2.0  # RedEdge for vitality
            
            region_features.append(np.concatenate([mean, std]))
            
        region_features = np.array(region_features)
        t_feat = perf_counter() - t_feat_start
        logger.info(f"PERF: Signature Extraction (means+text) took {t_feat:.2f}s for {len(unique_segs)} superpixels")
        
        # 5. Class discovery (Spectral & Textural grouping)
        t_kms_start = perf_counter()
        n_groups = min(len(unique_segs), max(2, request.granularity))
        logger.info(f"Grouping {len(unique_segs)} superpixels into {n_groups} spectral/texture classes...")
        
        # Using K-Means under the hood but logging generically as requested
        classifier = KMeans(n_clusters=n_groups, random_state=42, n_init="auto")
        region_labels = classifier.fit_predict(region_features)
        
        lookup = np.zeros(segments.max() + 1, dtype=np.int32)
        for sid, lbl in zip(unique_segs, region_labels):
            lookup[sid] = lbl
        labels_2d = lookup[segments]
        
        # 6. Vectorize
        sieved = sieve(labels_2d.astype(np.int32), 2, connectivity=8)
        results = (
            {'properties': {'object_class': int(v)}, 'geometry': s}
            for s, v in shapes(sieved, mask=(sieved >= 0), transform=data["transform"])
        )
        
        gdf = gpd.GeoDataFrame.from_features(list(results), crs=data["native_crs"])
        t_vec = time.perf_counter() - t_kms_start # Approximation if t_kms_start was the last anchor
        logger.info(f"PERF: Vectorization took {t_vec:.2f}s for {len(gdf)} polygons")
        
        gdf['polygon_id'] = range(len(gdf))
        gdf['class_id'] = -1
        gdf['label'] = 'unlabeled'
        gdf['area_sqm'] = gdf.geometry.area
        
        STATE["gdf"] = gdf.copy()
        gdf_wgs84 = gdf.to_crs("EPSG:4326")
        
        duration_sec = round(time.perf_counter() - t0, 1)
        return {
            "geojson": gdf_wgs84.__geo_interface__,
            "duration_sec": duration_sec
        }
    except Exception as e:
        logger.exception("Segmentation failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/save")
async def save_labels(request: SaveRequest):
    """
    Merges tool labels into the GeoDataFrame and saves to GeoParquet.
    """
    if STATE["gdf"] is None:
        raise HTTPException(status_code=400, detail="No active segmentation to save")
        
    try:
        # Merge polygon labels into our current GDF
        gdf = STATE["gdf"]
        
        # Update class_id and label based on polygon_id
        for poly_id_str, info in request.polygon_labels.items():
            mask = gdf['polygon_id'] == int(poly_id_str)
            gdf.loc[mask, 'class_id'] = info['class_id']
            gdf.loc[mask, 'label'] = info.get('label', info.get('name', 'unknown'))
            
        # Dynamic Project Directory: cwd/exports/{project_name}/labels
        base_dir = Path.cwd() / "exports"
        # Access project_name from the request object
        project_dir = base_dir / request.project_name / "labels"
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-naming: project_labels_YYYYMMDD_HHMM.gpkg
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        fname = f"{request.project_name}_labels_{ts}.gpkg"
        path = project_dir / fname

        # Filter to keep only labeled polygons
        export_gdf = gdf[gdf['class_id'] != -1].copy()
        
        if len(export_gdf) == 0:
            return {"status": "warning", "detail": "No polygons were labeled. Export skipped."}

        # Save to GeoPackage as requested (.gpkg)
        export_gdf.to_file(path, driver="GPKG", index=False)
        logger.info(f"Saved {len(export_gdf)} labeled polygons to {path}")
        
        return {"status": "success", "path": str(path.absolute()), "assigned_polygons": len(request.polygon_labels)}
        
    except Exception as e:
        logger.exception("Save failed")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
