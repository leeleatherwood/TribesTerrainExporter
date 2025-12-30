#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TribesTerrainExporter - Starsiege: Tribes Terrain File Exporter.

This tool exports Starsiege: Tribes .ted terrain files to heightmaps,
material maps, and rendered textures for use in modern game engines.

Features:
    * GUI mode with live preview
    * Command-line mode for batch processing
    * Heightmap export (16-bit PNG and RAW)
    * Material and flag map export
    * Full terrain texture rendering
    * DML material parsing

Example:
    GUI Mode::

        $ python TribesTerrainExporter.py

    Command-Line Mode::

        $ python TribesTerrainExporter.py terrain.ted
        $ python TribesTerrainExporter.py terrain.ted -o ./exports --tile-size 128 --debug

Note:
    Requires Python 3.7+ with Pillow and NumPy.

Shazbot! 🔥
"""
from __future__ import annotations

__version__ = "2025.12.30"
__author__ = "TribesToBlender Project"
__license__ = "GPL-3.0"

# ===== DISCLAIMER =====
# This script was vibe coded using GitHub Copilot (Claude Opus 4.5) by an
# author who freely admits to knowing absolutely nothing about Python.
# ======================

import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


# =============================================================================
# Dependency Installation
# =============================================================================

REQUIRED_PACKAGES = {
    'numpy': 'numpy',
    'PIL': 'Pillow',
}

# TribesToBlender repository settings
TRIBES_TO_BLENDER_REPO = "tekrog/TribesToBlender"
TRIBES_TO_BLENDER_BRANCH = "bovidi"
TRIBES_TO_BLENDER_ZIP_URL = f"https://github.com/{TRIBES_TO_BLENDER_REPO}/archive/refs/heads/{TRIBES_TO_BLENDER_BRANCH}.zip"
TRIBES_TO_BLENDER_DIR = "TribesToBlender"

# Get script directory
_SCRIPT_DIR = Path(__file__).parent.resolve()


def _install_package(package_name: str) -> bool:
    """
    Install a package using pip.
    
    Args:
        package_name: Name of the package to install.
    
    Returns:
        True if installation succeeded, False otherwise.
    """
    try:
        print(f"Installing {package_name}...")
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', package_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        print(f"  ✓ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"  ✗ Failed to install {package_name}")
        return False


def _ensure_dependencies() -> None:
    """
    Check for required dependencies and install any that are missing.
    """
    missing_packages = []
    
    for import_name, package_name in REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append((import_name, package_name))
    
    if missing_packages:
        print("=" * 50)
        print("Installing missing dependencies...")
        print("=" * 50)
        
        for import_name, package_name in missing_packages:
            _install_package(package_name)
        
        print("=" * 50)
        print()


def _download_tribes_to_blender() -> bool:
    """
    Download TribesToBlender repository from GitHub.
    
    Returns:
        True if download succeeded, False otherwise.
    """
    tribes_dir = _SCRIPT_DIR / TRIBES_TO_BLENDER_DIR
    
    # Check if already downloaded
    if tribes_dir.exists():
        # Verify key files exist
        terrain_file = tribes_dir / "terrain" / "terrain.py"
        dml_file = tribes_dir / "interior_shape_module" / "dml.py"
        if terrain_file.exists() and dml_file.exists():
            return True
    
    print("=" * 50)
    print("Downloading TribesToBlender from GitHub...")
    print(f"  Branch: {TRIBES_TO_BLENDER_BRANCH}")
    print("=" * 50)
    
    zip_path = _SCRIPT_DIR / "TribesToBlender.zip"
    
    try:
        # Download ZIP file
        print("  Downloading...")
        urllib.request.urlretrieve(TRIBES_TO_BLENDER_ZIP_URL, zip_path)
        
        # Extract ZIP file
        print("  Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(_SCRIPT_DIR)
        
        # Rename extracted folder (GitHub names it repo-branch)
        extracted_name = f"TribesToBlender-{TRIBES_TO_BLENDER_BRANCH}"
        extracted_dir = _SCRIPT_DIR / extracted_name
        
        if extracted_dir.exists():
            # Remove existing TribesToBlender dir if present
            if tribes_dir.exists():
                import shutil
                shutil.rmtree(tribes_dir)
            extracted_dir.rename(tribes_dir)
        
        # Clean up ZIP file
        zip_path.unlink()
        
        print(f"  ✓ TribesToBlender downloaded to: {tribes_dir}")
        print("=" * 50)
        print()
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to download TribesToBlender: {e}")
        print("    Please download manually from:")
        print(f"    {TRIBES_TO_BLENDER_ZIP_URL}")
        print("=" * 50)
        
        # Clean up partial download
        if zip_path.exists():
            zip_path.unlink()
        
        return False


def _ensure_tribes_to_blender() -> None:
    """
    Ensure TribesToBlender is available, downloading if necessary.
    """
    tribes_dir = _SCRIPT_DIR / TRIBES_TO_BLENDER_DIR
    
    # Add TribesToBlender to path if it exists
    if tribes_dir.exists():
        if str(tribes_dir) not in sys.path:
            sys.path.insert(0, str(tribes_dir))
        return
    
    # Try to download
    if _download_tribes_to_blender():
        if str(tribes_dir) not in sys.path:
            sys.path.insert(0, str(tribes_dir))


# Install dependencies before importing them
_ensure_dependencies()

# Ensure TribesToBlender is available
_ensure_tribes_to_blender()


# Standard library imports
import argparse
import logging
import os
import shutil
import tempfile
import traceback
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Type checking imports
if TYPE_CHECKING:
    from numpy.typing import NDArray

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Optional PIL import
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore
    logger.warning("PIL (Pillow) not available. Image export features will be limited.")

# Optional tkinter for GUI
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, Canvas, Scrollbar
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False


# =============================================================================
# Constants
# =============================================================================

__all__ = [
    'ExportConfig',
    'TerrainExporter',
    'export_terrain',
    'TerrainExportError',
]

# Directory paths (relative to script location)
SCRIPT_DIR = Path(__file__).parent.resolve()
TEXTURES_DIR = SCRIPT_DIR / "textures"
DML_DIR = SCRIPT_DIR / "dml"
TERRAIN_ARCHIVES_DIR = SCRIPT_DIR / "terrain_archives"

# =============================================================================
# UI Constants (matching Img2Splat exactly)
# =============================================================================

# Preview sizes
PREVIEW_SIZE = 512
LAYER_PREVIEW_SIZE = 1024

# Number of preview panels
NUM_PREVIEWS = 4
PREVIEW_LABELS = ["Terrain", "Heightmap", "Materials", "Flags"]
DEFAULT_PREVIEW_INDEX = 0  # Terrain Texture is default

# Valid tile sizes for dropdown
VALID_TILE_SIZES = [16, 32, 64, 128, 256]

# UI Messages
MSG_NO_TED = "Please load a .ted file first"
MSG_NO_DML = "Please select or load a DML file"
MSG_NO_EXPORTS = "Export terrain first"
MSG_SUCCESS_TITLE = "Success"
MSG_ERROR_TITLE = "Error"
MSG_GENERATING = "🏔️  Exporting terrain from {}"
MSG_SUCCESS_EXPORT = "✓ Terrain exported successfully"
MSG_SUCCESS_SAVED = "✓ Saved: {} ({}×{})"

# =============================================================================
# File Extension Constants
# =============================================================================

# File extensions
class FileExtensions:
    """File extension constants."""
    DML_ZIP = 'dml.zip'
    TERRAIN_ZIP = 'terrain.zip'
    PNG = '.png'
    DML = '.dml'
    BMP = '.bmp'
    DTF = '.dtf'
    DTB = '.dtb'
    TED = '.ted'
    RAW = '.raw'


# Numeric constants
class Defaults:
    """Default values for export settings."""
    TILE_SIZE = 64
    PREVIEW_MATERIAL_COUNT = 10
    PNG_QUALITY = 95
    TERRAIN_SIZE_METERS = 2048.0


# Tile transformation lookup table for material flags
# Maps flag value to list of PIL transpose operations
FLAG_TRANSFORMS: Dict[int, List[str]] = {
    0: [],
    1: ['rotate_270'],
    2: ['flip_h'],
    3: ['rotate_180'],
    4: ['flip_h', 'rotate_180'],
    5: ['flip_h', 'rotate_90'],
    6: ['rotate_180'],
    7: ['flip_h', 'rotate_90', 'flip_v'],
    8: [],
    9: ['flip_v'],
    10: ['flip_h'],
    11: ['flip_h', 'rotate_180'],
    12: ['rotate_270'],
    13: ['flip_h', 'rotate_90'],
    14: ['flip_h', 'rotate_270'],
    15: ['flip_h', 'rotate_90'],
    128: [],
    129: ['rotate_270'],
    130: ['flip_h'],
    131: ['rotate_180'],
    132: ['flip_h', 'rotate_180'],
    133: ['flip_v', 'rotate_270'],
    134: ['flip_h', 'flip_v'],
    135: ['rotate_90'],
    136: [],
    137: ['rotate_180'],
    138: [],
    139: ['rotate_180'],
    140: ['flip_h'],
    141: ['rotate_90'],
    142: ['flip_h'],
    143: ['rotate_90'],
}


# =============================================================================
# Exceptions
# =============================================================================

class TerrainExportError(Exception):
    """Base exception for terrain export errors."""
    pass


class FileNotFoundError(TerrainExportError):
    """Raised when a required file is not found."""
    pass


class ParseError(TerrainExportError):
    """Raised when a file cannot be parsed."""
    pass


class ExportError(TerrainExportError):
    """Raised when export operation fails."""
    pass


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExportConfig:
    """Configuration for terrain export operations.
    
    Attributes:
        tile_size: Size of each texture tile in pixels.
        debug: Enable debug mode (draw borders and flag values on tiles).
        output_dir: Output directory path. If None, uses <terrain_name>_export.
        textures_dir: Directory containing texture files.
        dml_dir: Directory containing DML files.
        terrain_archives_dir: Directory containing Terrain.zip files.
    """
    tile_size: int = Defaults.TILE_SIZE
    debug: bool = False
    output_dir: Optional[Path] = None
    textures_dir: Path = field(default_factory=lambda: TEXTURES_DIR)
    dml_dir: Path = field(default_factory=lambda: DML_DIR)
    terrain_archives_dir: Path = field(default_factory=lambda: TERRAIN_ARCHIVES_DIR)
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {self.tile_size}")
        
        # Convert string paths to Path objects
        if isinstance(self.textures_dir, str):
            self.textures_dir = Path(self.textures_dir)
        if isinstance(self.dml_dir, str):
            self.dml_dir = Path(self.dml_dir)
        if isinstance(self.terrain_archives_dir, str):
            self.terrain_archives_dir = Path(self.terrain_archives_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


@dataclass
class ExportResults:
    """Results from terrain export for GUI display.
    
    Attributes:
        terrain_texture: Rendered terrain texture image.
        heightmap_vis: Heightmap visualization (8-bit scaled).
        materials_vis: Materials visualization (8-bit scaled).
        flags_vis: Flags visualization (8-bit scaled).
        heightmap_16bit: Full 16-bit heightmap data.
        heightmap_raw: Raw heightmap bytes for Unity.
        materials_raw: Raw material indices.
        flags_raw: Raw material flags.
        metadata_text: Terrain metadata as text.
        stats: Export statistics dictionary.
    """
    terrain_texture: Optional[Any] = None  # PIL Image
    heightmap_vis: Optional[Any] = None    # PIL Image
    materials_vis: Optional[Any] = None    # PIL Image
    flags_vis: Optional[Any] = None        # PIL Image
    # Full data (for saving)
    heightmap_16bit: Optional[Any] = None  # numpy array
    heightmap_raw: Optional[bytes] = None
    materials_raw: Optional[Any] = None    # numpy array
    flags_raw: Optional[Any] = None        # numpy array
    metadata_text: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Module Import Setup
# =============================================================================

def _setup_imports() -> Tuple[Any, Any]:
    """
    Set up imports for terrain and DML modules.
    
    Returns:
        Tuple of (terrain_module, dml_module)
    
    Raises:
        ImportError: If required modules cannot be imported.
    """
    # Add TribesToBlender directory to path
    tribes_dir = SCRIPT_DIR / TRIBES_TO_BLENDER_DIR
    if tribes_dir.exists() and str(tribes_dir) not in sys.path:
        sys.path.insert(0, str(tribes_dir))
    
    # Also try parent directory (legacy support)
    parent_dir = SCRIPT_DIR.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    # Create fake modules for Blender-specific imports
    class _FakeModule:
        """Placeholder module for standalone use."""
        def __getattr__(self, name: str) -> Any:
            return _FakeModule()
    
    # Create fake bpy module (Blender Python API)
    if 'bpy' not in sys.modules:
        sys.modules['bpy'] = _FakeModule()  # type: ignore
    
    # Set up TribesToBlender module structure
    if 'TribesToBlender' not in sys.modules:
        import types
        tribes_module = types.ModuleType('TribesToBlender')
        tribes_module.__path__ = [str(tribes_dir)]  # type: ignore
        sys.modules['TribesToBlender'] = tribes_module
    
    try:
        # Import terrain modules
        from TribesToBlender.terrain import terrain as terrain_mod
        from TribesToBlender.interior_shape_module import dml as dml_mod
        
        # Return the modules (which contain the classes)
        return terrain_mod, dml_mod
        
    except ImportError as e:
        raise ImportError(
            f"Cannot import terrain modules. Ensure TribesToBlender is accessible: {e}"
        ) from e


# Initialize modules
try:
    terrain_module, dml_module = _setup_imports()
except ImportError as e:
    logger.error(str(e))
    terrain_module = None
    dml_module = None


# =============================================================================
# Helper Functions
# =============================================================================

def _decode_string(value: Optional[Union[bytes, str]]) -> Optional[str]:
    """
    Decode a bytes string to UTF-8 and strip null terminators.
    
    Args:
        value: Bytes string or regular string to decode.
    
    Returns:
        Decoded and cleaned string, or None if input is None/empty.
    """
    if not value:
        return None
    if isinstance(value, bytes):
        return value.decode('utf-8').strip('\x00')
    return str(value).strip('\x00')


def _sanitize_filename(name: str) -> str:
    """
    Sanitize a string for use as a filename.
    
    Args:
        name: Original filename string.
    
    Returns:
        Sanitized filename with only alphanumeric, dash, and underscore chars.
    """
    return "".join(c for c in name if c.isalnum() or c in ('-', '_'))


def _scale_for_visualization(data: NDArray) -> NDArray:
    """
    Scale data to use full 0-255 range for better visualization.
    
    Args:
        data: Input numpy array.
    
    Returns:
        Scaled array as uint8.
    """
    unique_vals = np.unique(data)
    if len(unique_vals) <= 1:
        return data.astype(np.uint8)
    
    scaled = np.zeros_like(data, dtype=np.uint8)
    for idx, val in enumerate(unique_vals):
        mapped_value = int((idx / (len(unique_vals) - 1)) * 255)
        scaled[data == val] = mapped_value
    
    return scaled


# =============================================================================
# Archive Extraction
# =============================================================================

class ArchiveExtractor:
    """Handles extraction of DML and Terrain archives."""
    
    def __init__(self, config: ExportConfig) -> None:
        """
        Initialize the archive extractor.
        
        Args:
            config: Export configuration.
        """
        self.config = config
    
    def extract_files_from_zip(
        self, 
        zip_path: Path, 
        output_dir: Path, 
        file_extension: str
    ) -> int:
        """
        Extract files with specific extension from a ZIP archive.
        
        Args:
            zip_path: Path to the ZIP file.
            output_dir: Directory to extract files to.
            file_extension: File extension to filter (e.g., '.png').
        
        Returns:
            Number of files extracted.
        
        Raises:
            ExportError: If extraction fails.
        """
        count = 0
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_info in zip_ref.filelist:
                    if file_info.filename.lower().endswith(file_extension.lower()):
                        filename = os.path.basename(file_info.filename)
                        if filename:  # Skip directory entries
                            file_data = zip_ref.read(file_info.filename)
                            output_path = output_dir / filename
                            output_path.write_bytes(file_data)
                            count += 1
        except zipfile.BadZipFile as e:
            raise ExportError(f"Invalid ZIP file {zip_path.name}: {e}") from e
        except OSError as e:
            raise ExportError(f"Failed to extract from {zip_path.name}: {e}") from e
        
        return count
    
    def extract_dml_textures(self) -> Path:
        """
        Extract PNG textures from *DML.zip files in the dml/ directory.
        
        Returns:
            Path to the textures directory.
        """
        logger.info("=== Extracting DML Archives ===")
        
        self.config.textures_dir.mkdir(parents=True, exist_ok=True)
        self.config.dml_dir.mkdir(parents=True, exist_ok=True)
        
        dml_zips = list(self.config.dml_dir.glob(f"*{FileExtensions.DML_ZIP}"))
        
        if not dml_zips:
            logger.warning(f"No DML.zip files found in {self.config.dml_dir}")
            return self.config.textures_dir
        
        logger.info(f"Found {len(dml_zips)} DML archives: {', '.join(z.name for z in dml_zips)}")
        
        for dml_zip in dml_zips:
            logger.info(f"Extracting PNG textures from {dml_zip.name}")
            try:
                png_count = self.extract_files_from_zip(
                    dml_zip, 
                    self.config.textures_dir, 
                    FileExtensions.PNG
                )
                logger.info(f"  Extracted {png_count} PNG textures")
            except ExportError as e:
                logger.error(f"  {e}")
        
        all_textures = list(self.config.textures_dir.glob(f"*{FileExtensions.PNG}"))
        logger.info(f"=== Total: {len(all_textures)} textures in {self.config.textures_dir} ===")
        
        return self.config.textures_dir
    
    def extract_terrain_dmls(self) -> Path:
        """
        Extract .dml files from *Terrain.zip archives.
        
        Returns:
            Path to the DML directory.
        """
        logger.info("=== Extracting DML Files ===")
        
        self.config.dml_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.config.terrain_archives_dir.exists():
            logger.warning(f"No terrain_archives/ directory found at {self.config.terrain_archives_dir}")
            return self.config.dml_dir
        
        terrain_zips = list(self.config.terrain_archives_dir.glob(f"*{FileExtensions.TERRAIN_ZIP}"))
        
        if not terrain_zips:
            logger.warning(f"No Terrain.zip files found in {self.config.terrain_archives_dir}")
            return self.config.dml_dir
        
        logger.info(f"Found {len(terrain_zips)} Terrain archives: {', '.join(z.name for z in terrain_zips)}")
        
        for terrain_zip in terrain_zips:
            logger.info(f"Extracting DML files from {terrain_zip.name}")
            try:
                dml_count = self.extract_files_from_zip(
                    terrain_zip, 
                    self.config.dml_dir, 
                    FileExtensions.DML
                )
                logger.info(f"  Extracted {dml_count} DML files")
            except ExportError as e:
                logger.error(f"  {e}")
        
        all_dmls = list(self.config.dml_dir.glob(f"*{FileExtensions.DML}"))
        logger.info(f"=== Total: {len(all_dmls)} DML files in {self.config.dml_dir} ===")
        
        return self.config.dml_dir


# =============================================================================
# DML Parser
# =============================================================================

@dataclass
class MaterialInfo:
    """Information about a single material."""
    index: int
    texture: Optional[str]
    exists: bool


class DMLParser:
    """Parses DML files and creates material mappings."""
    
    def __init__(self, config: ExportConfig) -> None:
        """
        Initialize the DML parser.
        
        Args:
            config: Export configuration.
        """
        self.config = config
    
    def parse(self, dml_path: Path) -> Tuple[Dict[int, Optional[str]], List[MaterialInfo]]:
        """
        Parse a DML file and create a mapping of material index to texture filename.
        
        Args:
            dml_path: Path to the .dml file.
        
        Returns:
            Tuple of:
                - dict: Mapping of material index -> texture filename (or None).
                - list: List of MaterialInfo objects.
        
        Raises:
            ParseError: If the DML file cannot be parsed.
        """
        logger.info(f"=== Parsing DML File: {dml_path.name} ===")
        
        if dml_module is None:
            raise ParseError("DML module not available")
        
        try:
            dml = dml_module.dml()
            dml.load_file(str(dml_path))
            
            logger.info(f"DML contains {len(dml.materials)} materials")
            
            material_map: Dict[int, Optional[str]] = {}
            material_details: List[MaterialInfo] = []
            
            available_textures = self._get_available_textures()
            
            for idx, material in enumerate(dml.materials):
                texture_name = self._extract_texture_name(material)
                texture_name = self._normalize_texture_name(texture_name)
                texture_exists = texture_name is not None and texture_name in available_textures
                
                material_map[idx] = texture_name if texture_exists else None
                material_details.append(MaterialInfo(
                    index=idx,
                    texture=texture_name,
                    exists=texture_exists
                ))
            
            self._log_material_summary(material_details)
            
            return material_map, material_details
            
        except Exception as e:
            raise ParseError(f"Error parsing DML file: {e}") from e
    
    def _get_available_textures(self) -> set:
        """Get set of available texture filenames (lowercase)."""
        if not self.config.textures_dir.exists():
            return set()
        return {f.name.lower() for f in self.config.textures_dir.glob(f"*{FileExtensions.PNG}")}
    
    def _extract_texture_name(self, material: Any) -> Optional[str]:
        """Extract texture name from a material object."""
        for attr in ('filename', 'texture', 'name'):
            if hasattr(material, attr):
                value = getattr(material, attr)
                if value:
                    return _decode_string(value)
        return None
    
    def _normalize_texture_name(self, texture_name: Optional[str]) -> Optional[str]:
        """Normalize texture name to lowercase .png format."""
        if not texture_name:
            return None
        
        texture_name = texture_name.strip()
        
        # Remove .BMP extension if present
        if texture_name.upper().endswith(FileExtensions.BMP.upper()):
            texture_name = texture_name[:-4]
        
        # Add .png extension
        if not texture_name.lower().endswith(FileExtensions.PNG):
            texture_name = texture_name.lower() + FileExtensions.PNG
        else:
            texture_name = texture_name.lower()
        
        return texture_name
    
    def _log_material_summary(self, materials: List[MaterialInfo]) -> None:
        """Log summary of parsed materials."""
        found_count = sum(1 for m in materials if m.exists)
        logger.info(f"  Found {found_count}/{len(materials)} textures in textures directory")
        
        preview_count = min(Defaults.PREVIEW_MATERIAL_COUNT, len(materials))
        logger.info(f"First {preview_count} material mappings:")
        
        for i in range(preview_count):
            m = materials[i]
            status = "✓" if m.exists else "✗"
            logger.info(f"  [{i}] {status} {m.texture}")
        
        if len(materials) > Defaults.PREVIEW_MATERIAL_COUNT:
            logger.info(f"  ... and {len(materials) - Defaults.PREVIEW_MATERIAL_COUNT} more")


# =============================================================================
# Image Export
# =============================================================================

class ImageExporter:
    """Handles exporting images in various formats."""
    
    @staticmethod
    def save_16bit_png(data: NDArray, output_path: Path, label: str) -> bool:
        """
        Save numpy array as 16-bit PNG image.
        
        Args:
            data: 2D numpy array of height data.
            output_path: Path to save the image.
            label: Label for logging.
        
        Returns:
            True if successful, False otherwise.
        """
        if not PIL_AVAILABLE:
            logger.warning(f"PIL required to export {label}")
            return False
        
        try:
            data_uint16 = data.astype(np.uint16)
            # Create 16-bit grayscale image (I;16 mode)
            img = Image.new('I;16', (data_uint16.shape[1], data_uint16.shape[0]))
            img.putdata(data_uint16.flatten().tolist())
            img.save(output_path)
            logger.info(f"{label} exported to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving {label}: {e}")
            return False
    
    @staticmethod
    def save_8bit_bmp(data: NDArray, output_path: Path, label: str) -> bool:
        """
        Save numpy array as 8-bit BMP image.
        
        Args:
            data: 2D numpy array.
            output_path: Path to save the image.
            label: Label for logging.
        
        Returns:
            True if successful, False otherwise.
        """
        if not PIL_AVAILABLE:
            logger.warning(f"PIL required to export {label}")
            return False
        
        try:
            # Create 8-bit grayscale image (L mode)
            img = Image.fromarray(data.astype(np.uint8))
            img = img.convert('L')
            img.save(output_path)
            logger.info(f"{label} exported to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving {label}: {e}")
            return False
    
    @staticmethod
    def save_raw_heightmap(data: NDArray, output_path: Path, label: str) -> bool:
        """
        Save heightmap as Unity-compatible RAW file (16-bit little-endian).
        
        Args:
            data: 2D numpy array of height data.
            output_path: Path to save the file.
            label: Label for logging.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            heightmap_uint16 = data.astype(np.uint16)
            with open(output_path, 'wb') as f:
                heightmap_uint16.tofile(f)
            
            logger.info(f"{label} exported to: {output_path}")
            logger.info(f"  Dimensions: {data.shape[1]}x{data.shape[0]} (width x height)")
            logger.info(f"  Format: 16-bit little-endian")
            return True
        except Exception as e:
            logger.error(f"Error saving {label}: {e}")
            return False


# =============================================================================
# Terrain Texture Renderer
# =============================================================================

class TerrainRenderer:
    """Renders terrain textures from material data."""
    
    def __init__(self, config: ExportConfig) -> None:
        """
        Initialize the terrain renderer.
        
        Args:
            config: Export configuration.
        """
        self.config = config
        self._texture_cache: Dict[str, Any] = {}
    
    def render(
        self, 
        terrain_obj: Any, 
        material_map: Dict[int, Optional[str]], 
        output_path: Path
    ) -> bool:
        """
        Render the full terrain texture.
        
        Args:
            terrain_obj: The terrain object with material data.
            material_map: Dictionary mapping material index -> texture filename.
            output_path: Where to save the rendered terrain texture.
        
        Returns:
            True if successful, False otherwise.
        """
        logger.info("=== Rendering Terrain Texture ===")
        
        if not PIL_AVAILABLE:
            logger.error("PIL (Pillow) is required for terrain texture rendering")
            return False
        
        try:
            mat_index = terrain_obj.get_material_map_image()
            mat_flags = terrain_obj.get_material_flags_image()
            
            height, width = mat_index.shape
            tile_size = self.config.tile_size
            
            logger.info(f"Terrain size: {width}x{height} squares")
            logger.info(f"Tile size: {tile_size}x{tile_size} pixels")
            logger.info(f"Output texture size: {width * tile_size}x{height * tile_size} pixels")
            
            output_img = Image.new('RGB', (width * tile_size, height * tile_size), color=(0, 0, 0))
            
            missing_textures: set = set()
            processed_count = 0
            
            for y in range(height):
                for x in range(width):
                    if self._process_tile(
                        x, y, 
                        mat_index[y, x], 
                        mat_flags[y, x],
                        material_map, 
                        output_img, 
                        missing_textures
                    ):
                        processed_count += 1
            
            self._log_render_results(processed_count, width * height, missing_textures)
            
            output_img.save(output_path, quality=Defaults.PNG_QUALITY)
            logger.info(f"Terrain texture saved to: {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error rendering terrain texture: {e}")
            traceback.print_exc()
            return False
    
    def render_to_image(
        self,
        terrain_obj: Any,
        material_map: Dict[int, Optional[str]],
        progress_callback: Optional[callable] = None
    ) -> Optional[Any]:
        """
        Render terrain texture to PIL Image (for GUI display).
        
        Args:
            terrain_obj: The terrain object with material data.
            material_map: Dictionary mapping material index -> texture filename.
            progress_callback: Optional function(percent, message) for progress updates.
        
        Returns:
            PIL Image of rendered terrain, or None on failure.
        """
        if not PIL_AVAILABLE:
            logger.error("PIL (Pillow) is required for terrain texture rendering")
            return None
        
        try:
            mat_index = terrain_obj.get_material_map_image()
            mat_flags = terrain_obj.get_material_flags_image()
            
            height, width = mat_index.shape
            tile_size = self.config.tile_size
            
            output_img = Image.new('RGB', (width * tile_size, height * tile_size), color=(0, 0, 0))
            
            missing_textures: set = set()
            processed_count = 0
            total_tiles = width * height
            
            for y in range(height):
                for x in range(width):
                    if self._process_tile(
                        x, y,
                        mat_index[y, x],
                        mat_flags[y, x],
                        material_map,
                        output_img,
                        missing_textures
                    ):
                        processed_count += 1
                
                # Progress callback every 10 rows
                if progress_callback and y % 10 == 0:
                    percent = 65 + (y / height) * 30  # 65% to 95%
                    progress_callback(percent, f"Rendering tiles: row {y + 1}/{height}")
            
            if progress_callback:
                progress_callback(95, "Terrain texture complete")
            
            return output_img
            
        except Exception as e:
            logger.error(f"Error rendering terrain texture: {e}")
            traceback.print_exc()
            return None
    
    def _process_tile(
        self,
        x: int,
        y: int,
        mat_idx: int,
        flags: int,
        material_map: Dict[int, Optional[str]],
        output_img: Any,
        missing_textures: set
    ) -> bool:
        """Process a single tile and paste it into the output image."""
        texture_name = material_map.get(mat_idx)
        
        if texture_name is None:
            missing_textures.add(mat_idx)
            return False
        
        tile_img = self._get_texture(texture_name)
        if tile_img is None:
            missing_textures.add(mat_idx)
            return False
        
        transformed_tile = self._apply_transforms(tile_img, flags)
        
        if self.config.debug:
            transformed_tile = self._draw_debug_overlay(transformed_tile, flags)
        
        paste_x = x * self.config.tile_size
        paste_y = y * self.config.tile_size
        output_img.paste(transformed_tile, (paste_x, paste_y))
        
        return True
    
    def _get_texture(self, texture_name: str) -> Optional[Any]:
        """Load texture from cache or disk."""
        if texture_name in self._texture_cache:
            return self._texture_cache[texture_name]
        
        texture_path = self.config.textures_dir / texture_name
        if not texture_path.exists():
            return None
        
        try:
            tile_img = Image.open(texture_path).convert('RGB')
            if tile_img.size != (self.config.tile_size, self.config.tile_size):
                tile_img = tile_img.resize(
                    (self.config.tile_size, self.config.tile_size), 
                    Image.Resampling.LANCZOS
                )
            self._texture_cache[texture_name] = tile_img
            return tile_img
        except Exception as e:
            logger.error(f"Error loading texture {texture_name}: {e}")
            return None
    
    def _apply_transforms(self, tile_img: Any, flags: int) -> Any:
        """Apply transformations based on material flags."""
        transformed = tile_img.copy()
        operations = FLAG_TRANSFORMS.get(flags, [])
        
        transform_map = {
            'flip_h': Image.Transpose.FLIP_LEFT_RIGHT,
            'flip_v': Image.Transpose.FLIP_TOP_BOTTOM,
            'rotate_90': Image.Transpose.ROTATE_90,
            'rotate_180': Image.Transpose.ROTATE_180,
            'rotate_270': Image.Transpose.ROTATE_270,
        }
        
        for operation in operations:
            if operation in transform_map:
                transformed = transformed.transpose(transform_map[operation])
        
        return transformed
    
    def _draw_debug_overlay(self, tile: Any, flags: int) -> Any:
        """Draw debug overlay on a tile (border and flag number)."""
        tile_size = self.config.tile_size
        draw = ImageDraw.Draw(tile)
        
        # Draw border
        draw.rectangle([(0, 0), (tile_size-1, tile_size-1)], outline=(0, 0, 0), width=1)
        
        # Add flag value text
        flag_text = str(flags)
        try:
            font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), flag_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except Exception:
            font = None
            text_width = len(flag_text) * 6
            text_height = 8
        
        text_x = (tile_size - text_width) // 2
        text_y = (tile_size - text_height) // 2
        
        # Draw text outline
        for ox in [-1, 0, 1]:
            for oy in [-1, 0, 1]:
                if ox != 0 or oy != 0:
                    draw.text((text_x + ox, text_y + oy), flag_text, fill=(0, 0, 0), font=font)
        
        draw.text((text_x, text_y), flag_text, fill=(255, 255, 255), font=font)
        
        return tile
    
    def _log_render_results(
        self, 
        processed: int, 
        total: int, 
        missing: set
    ) -> None:
        """Log rendering results."""
        percent = 100.0 * processed / total if total > 0 else 0
        logger.info(f"Processed: {processed}/{total} squares ({percent:.1f}%)")
        
        if missing:
            logger.warning(f"{len(missing)} material indices had missing textures")


# =============================================================================
# Metadata Export
# =============================================================================

class MetadataExporter:
    """Exports terrain metadata to text files."""
    
    @staticmethod
    def export(terrain_obj: Any, output_path: Path) -> bool:
        """
        Save terrain metadata to a text file.
        
        Args:
            terrain_obj: The terrain object.
            output_path: Path to save the metadata file.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            ter = terrain_obj
            terrain_size = Defaults.TERRAIN_SIZE_METERS
            meters_per_quad = terrain_size / ter.size_x
            actual_min, actual_max = ter.get_true_height_range()
            height_span = actual_max - actual_min
            
            lines = [
                "Starsiege: Tribes Terrain Metadata",
                "=" * 50,
                "",
                f"Name: {_decode_string(ter.name_id) or 'Unknown'}",
                f"DML: {_decode_string(ter.dml_name) or 'Unknown'}",
                "",
                "DIMENSIONS",
                "-" * 50,
                f"Heightmap resolution: {ter.size_x + 1} x {ter.size_y + 1} pixels",
                f"Terrain quads: {ter.size_x} x {ter.size_y}",
                f"Meters per quad: {meters_per_quad:.2f}m",
                f"Total terrain size: {terrain_size:.0f}m x {terrain_size:.0f}m",
                f"DFL grid size: {ter.dfl_size_x} x {ter.dfl_size_y}",
                f"Scale value: {ter.scale}",
                "",
                "UNITY TERRAIN IMPORT SETTINGS",
                "-" * 50,
                f"Terrain Width: {terrain_size:.0f}",
                f"Terrain Length: {terrain_size:.0f}",
                f"Terrain Height: {height_span:.2f}",
                f"Heightmap Resolution: {ter.size_x + 1}",
                "",
                "HEIGHT INFORMATION",
                "-" * 50,
                f"Height range (GBLK): {ter.height_fmin:.2f} to {ter.height_fmax:.2f}",
                f"Height range (DTF): {ter.height_range_min:.2f} to {ter.height_range_max:.2f}",
                f"Actual height range: {actual_min:.2f} to {actual_max:.2f}",
                f"Height span: {height_span:.2f} meters",
                "",
                "BOUNDING BOX",
                "-" * 50,
                f"Origin: ({ter.origin_x}, {ter.origin_y})",
                f"Bottom-Left: {ter.bounds_bl}",
                f"Top-Right: {ter.bounds_tr}",
                "",
                "ADDITIONAL INFO",
                "-" * 50,
                f"Detail count: {ter.detail_count}",
                f"Light scale: {ter.light_scale}",
                f"Block pattern: {ter.dfl_block_pattern}",
                f"Last block ID: {ter.last_block_id}",
            ]
            
            if ter.block_map:
                lines.extend(["", "BLOCK MAP", "-" * 50, f"Block map: {ter.block_map}"])
            
            output_path.write_text("\n".join(lines))
            logger.info(f"Terrain metadata exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving terrain metadata: {e}")
            return False


# =============================================================================
# Main Exporter Class
# =============================================================================

class TerrainExporter:
    """Main class for exporting Tribes terrain files."""
    
    def __init__(self, config: Optional[ExportConfig] = None) -> None:
        """
        Initialize the terrain exporter.
        
        Args:
            config: Export configuration. Uses defaults if not provided.
        """
        self.config = config or ExportConfig()
        self._temp_dir: Optional[Path] = None
    
    def export(self, terrain_file: Path) -> bool:
        """
        Export terrain from a .ted file.
        
        Args:
            terrain_file: Path to the .ted terrain file.
        
        Returns:
            True if export was successful, False otherwise.
        """
        if terrain_module is None:
            logger.error("Terrain module not available. Cannot export.")
            return False
        
        terrain_file = Path(terrain_file)
        if not terrain_file.exists():
            logger.error(f"File not found: {terrain_file}")
            return False
        
        try:
            # Extract archives
            extractor = ArchiveExtractor(self.config)
            extractor.extract_dml_textures()
            extractor.extract_terrain_dmls()
            
            # Load terrain
            terrain_obj = self._load_terrain(terrain_file)
            if terrain_obj is None:
                return False
            
            terrain_obj.print_stats()
            
            # Parse DML
            material_map = self._parse_terrain_dml(terrain_obj)
            
            # Setup output directory
            output_dir = self._setup_output_dir(terrain_file)
            terrain_name = _sanitize_filename(terrain_file.stem)
            
            # Export all data
            self._export_heightmap(terrain_obj, output_dir, terrain_name)
            self._export_material_maps(terrain_obj, output_dir, terrain_name)
            self._export_metadata(terrain_obj, output_dir, terrain_name)
            
            if material_map:
                self._render_terrain_texture(terrain_obj, material_map, output_dir, terrain_name)
            
            logger.info(f"\nExport complete! Files saved to: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            traceback.print_exc()
            return False
        finally:
            self._cleanup()
    
    def _load_terrain(self, terrain_file: Path) -> Optional[Any]:
        """Load terrain from file, handling ZIP archives."""
        dtf_file = None
        dtb_file = None
        
        try:
            if zipfile.is_zipfile(terrain_file):
                logger.info("Terrain file is a ZIP archive, extracting...")
                self._temp_dir = Path(tempfile.mkdtemp())
                
                with zipfile.ZipFile(terrain_file, 'r') as zip_ref:
                    zip_ref.extractall(self._temp_dir)
                
                for path in self._temp_dir.rglob('*'):
                    if path.suffix.lower() == FileExtensions.DTF:
                        dtf_file = path
                    elif path.suffix.lower() == FileExtensions.DTB:
                        dtb_file = path
                
                if dtf_file and dtb_file:
                    logger.info(f"Found terrain files: {dtf_file.name} and {dtb_file.name}")
        except Exception as e:
            logger.error(f"Error checking/extracting terrain ZIP: {e}")
        
        ter = terrain_module.terrain()
        
        if dtf_file and dtb_file:
            ter.load_dtf_binary(dtf_file.read_bytes())
            ter.load_binary(dtb_file.read_bytes())
        else:
            ter.load_file(str(terrain_file))
        
        return ter
    
    def _parse_terrain_dml(self, terrain_obj: Any) -> Dict[int, Optional[str]]:
        """Parse the DML file referenced by the terrain."""
        dml_name = _decode_string(terrain_obj.dml_name)
        
        if not dml_name:
            logger.warning("No DML file referenced in terrain")
            return {}
        
        dml_path = self.config.dml_dir / dml_name
        
        if not dml_path.exists():
            logger.warning(f"DML file not found: {dml_path}")
            return {}
        
        try:
            parser = DMLParser(self.config)
            material_map, _ = parser.parse(dml_path)
            return material_map
        except ParseError as e:
            logger.error(str(e))
            return {}
    
    def _setup_output_dir(self, terrain_file: Path) -> Path:
        """Create and return the output directory."""
        if self.config.output_dir:
            output_dir = self.config.output_dir
        else:
            output_dir = SCRIPT_DIR / f"{terrain_file.stem}_export"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def _export_heightmap(self, terrain_obj: Any, output_dir: Path, name: str) -> None:
        """Export heightmap in PNG and RAW formats."""
        heightmap = terrain_obj.get_heightmap_0_65535()
        
        ImageExporter.save_16bit_png(
            heightmap, 
            output_dir / f"{name}_heightmap.png", 
            "Heightmap"
        )
        
        ImageExporter.save_raw_heightmap(
            heightmap, 
            output_dir / f"{name}_heightmap.raw", 
            "Heightmap RAW"
        )
    
    def _export_material_maps(self, terrain_obj: Any, output_dir: Path, name: str) -> None:
        """Export material index and flags maps."""
        if terrain_obj.mat_index is not None and terrain_obj.mat_index.size > 0:
            mat_map = terrain_obj.get_material_map_image()
            
            logger.info("Material Index Stats:")
            logger.info(f"  Min: {np.min(mat_map)}, Max: {np.max(mat_map)}")
            logger.info(f"  Unique values: {len(np.unique(mat_map))}")
            
            ImageExporter.save_8bit_bmp(mat_map, output_dir / f"{name}_materials_raw.bmp", "Material map (raw)")
            ImageExporter.save_8bit_bmp(
                _scale_for_visualization(mat_map), 
                output_dir / f"{name}_materials_vis.bmp", 
                "Material map (vis)"
            )
        
        if terrain_obj.mat_flags is not None and terrain_obj.mat_flags.size > 0:
            flags_map = terrain_obj.get_material_flags_image()
            
            logger.info("Material Flags Stats:")
            logger.info(f"  Min: {np.min(flags_map)}, Max: {np.max(flags_map)}")
            logger.info(f"  Unique values: {len(np.unique(flags_map))}")
            
            ImageExporter.save_8bit_bmp(flags_map, output_dir / f"{name}_material_flags_raw.bmp", "Material flags (raw)")
            ImageExporter.save_8bit_bmp(
                _scale_for_visualization(flags_map), 
                output_dir / f"{name}_material_flags_vis.bmp", 
                "Material flags (vis)"
            )
    
    def _export_metadata(self, terrain_obj: Any, output_dir: Path, name: str) -> None:
        """Export terrain metadata."""
        MetadataExporter.export(terrain_obj, output_dir / f"{name}_meta.txt")
    
    def _render_terrain_texture(
        self, 
        terrain_obj: Any, 
        material_map: Dict[int, Optional[str]], 
        output_dir: Path, 
        name: str
    ) -> None:
        """Render the full terrain texture."""
        renderer = TerrainRenderer(self.config)
        renderer.render(terrain_obj, material_map, output_dir / f"{name}_terrain_texture.png")
    
    def export_to_memory(
        self,
        terrain_file: Path,
        progress_callback: Optional[callable] = None
    ) -> Optional[ExportResults]:
        """
        Export terrain to memory for GUI display.
        
        Args:
            terrain_file: Path to the .ted terrain file.
            progress_callback: Optional function(percent, message) for progress updates.
        
        Returns:
            ExportResults object with all generated images, or None on failure.
        """
        if terrain_module is None:
            logger.error("Terrain module not available. Cannot export.")
            return None
        
        terrain_file = Path(terrain_file)
        if not terrain_file.exists():
            logger.error(f"File not found: {terrain_file}")
            return None
        
        results = ExportResults()
        
        try:
            if progress_callback:
                progress_callback(5, "Extracting archives...")
            
            # Extract archives
            extractor = ArchiveExtractor(self.config)
            extractor.extract_dml_textures()
            extractor.extract_terrain_dmls()
            
            if progress_callback:
                progress_callback(15, "Loading terrain...")
            
            # Load terrain
            terrain_obj = self._load_terrain(terrain_file)
            if terrain_obj is None:
                return None
            
            if progress_callback:
                progress_callback(25, "Parsing DML...")
            
            # Parse DML
            material_map = self._parse_terrain_dml(terrain_obj)
            
            if progress_callback:
                progress_callback(35, "Generating heightmap...")
            
            # Generate heightmap
            heightmap = terrain_obj.get_heightmap_0_65535()
            results.heightmap_16bit = heightmap
            results.heightmap_raw = heightmap.astype(np.uint16).tobytes()
            
            # Create heightmap visualization (scale to 8-bit)
            if PIL_AVAILABLE:
                heightmap_8bit = (heightmap / 256).astype(np.uint8)
                results.heightmap_vis = Image.fromarray(heightmap_8bit)
            
            if progress_callback:
                progress_callback(45, "Generating material maps...")
            
            # Generate material maps
            if terrain_obj.mat_index is not None and terrain_obj.mat_index.size > 0:
                mat_map = terrain_obj.get_material_map_image()
                results.materials_raw = mat_map
                
                if PIL_AVAILABLE:
                    mat_vis = _scale_for_visualization(mat_map)
                    results.materials_vis = Image.fromarray(mat_vis)
            
            if progress_callback:
                progress_callback(55, "Generating flags maps...")
            
            # Generate flags maps
            if terrain_obj.mat_flags is not None and terrain_obj.mat_flags.size > 0:
                flags_map = terrain_obj.get_material_flags_image()
                results.flags_raw = flags_map
                
                if PIL_AVAILABLE:
                    flags_vis = _scale_for_visualization(flags_map)
                    results.flags_vis = Image.fromarray(flags_vis)
            
            if progress_callback:
                progress_callback(60, "Generating metadata...")
            
            # Generate metadata text
            results.metadata_text = self._generate_metadata_text(terrain_obj)
            
            if progress_callback:
                progress_callback(65, "Rendering terrain texture...")
            
            # Render terrain texture
            if material_map and PIL_AVAILABLE:
                renderer = TerrainRenderer(self.config)
                results.terrain_texture = renderer.render_to_image(
                    terrain_obj, material_map, progress_callback
                )
            
            if progress_callback:
                progress_callback(100, "Complete!")
            
            # Store stats
            results.stats = {
                'terrain_name': _decode_string(terrain_obj.name_id) or terrain_file.stem,
                'dml_name': _decode_string(terrain_obj.dml_name) or 'Unknown',
                'size_x': terrain_obj.size_x,
                'size_y': terrain_obj.size_y,
                'heightmap_size': f"{terrain_obj.size_x + 1}×{terrain_obj.size_y + 1}",
                'material_count': len(material_map) if material_map else 0,
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            traceback.print_exc()
            return None
        finally:
            self._cleanup()
    
    def _generate_metadata_text(self, terrain_obj: Any) -> str:
        """Generate metadata text for a terrain object."""
        try:
            ter = terrain_obj
            terrain_size = Defaults.TERRAIN_SIZE_METERS
            meters_per_quad = terrain_size / ter.size_x
            actual_min, actual_max = ter.get_true_height_range()
            height_span = actual_max - actual_min
            
            lines = [
                "Starsiege: Tribes Terrain Metadata",
                "=" * 50,
                "",
                f"Name: {_decode_string(ter.name_id) or 'Unknown'}",
                f"DML: {_decode_string(ter.dml_name) or 'Unknown'}",
                "",
                "DIMENSIONS",
                "-" * 50,
                f"Heightmap resolution: {ter.size_x + 1} x {ter.size_y + 1} pixels",
                f"Terrain quads: {ter.size_x} x {ter.size_y}",
                f"Meters per quad: {meters_per_quad:.2f}m",
                f"Total terrain size: {terrain_size:.0f}m x {terrain_size:.0f}m",
                "",
                "HEIGHT INFORMATION",
                "-" * 50,
                f"Actual height range: {actual_min:.2f} to {actual_max:.2f}",
                f"Height span: {height_span:.2f} meters",
            ]
            return "\n".join(lines)
        except Exception:
            return "Error generating metadata"
    
    def _cleanup(self) -> None:
        """Clean up temporary files."""
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None


# =============================================================================
# Public API
# =============================================================================

def export_terrain(
    filename: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    tile_size: int = Defaults.TILE_SIZE,
    debug: bool = False
) -> bool:
    """
    Export terrain from a .ted file.
    
    This is the main entry point for programmatic use.
    
    Args:
        filename: Path to .ted terrain file.
        output_dir: Output directory (default: ./<terrain_name>_export).
        tile_size: Size of each texture tile in pixels.
        debug: Enable debug mode (draw borders and flag values on tiles).
    
    Returns:
        True if export was successful, False otherwise.
    
    Example:
        >>> export_terrain("Raindance.ted", tile_size=128)
        True
    """
    config = ExportConfig(
        tile_size=tile_size,
        debug=debug,
        output_dir=Path(output_dir) if output_dir else None
    )
    
    exporter = TerrainExporter(config)
    return exporter.export(Path(filename))


# =============================================================================
# GUI Classes
# =============================================================================

if TKINTER_AVAILABLE:
    from PIL import ImageTk
    
    class FullSizeViewer(tk.Toplevel):
        """Full-size image viewer window with scrollbars.
        
        A modal dialog that displays an image at full resolution with
        vertical and horizontal scrollbars for navigation.
        
        Args:
            parent: Parent tkinter window
            image: PIL Image to display
            title: Window title (default: "Full Size View")
        """
        
        def __init__(
            self,
            parent: tk.Tk,
            image: Any,
            title: str = "Full Size View"
        ) -> None:
            """Initialize the full-size viewer window."""
            super().__init__(parent)
            self.title(title)
            self.image = image
            
            # Set window size (max 80% of screen)
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            window_width = min(int(screen_width * 0.8), image.width + 40)
            window_height = min(int(screen_height * 0.8), image.height + 40)
            
            self.geometry(f"{window_width}x{window_height}")
            
            # Create scrollable canvas
            frame = ttk.Frame(self)
            frame.pack(fill=tk.BOTH, expand=True)
            
            # Scrollbars
            v_scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL)
            h_scrollbar = ttk.Scrollbar(frame, orient=tk.HORIZONTAL)
            
            # Canvas
            self.canvas = Canvas(frame,
                                xscrollcommand=h_scrollbar.set,
                                yscrollcommand=v_scrollbar.set)
            
            v_scrollbar.config(command=self.canvas.yview)
            h_scrollbar.config(command=self.canvas.xview)
            
            # Layout
            v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Display image
            self.photo = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
            
            # Close button
            btn_frame = ttk.Frame(self)
            btn_frame.pack(side=tk.BOTTOM, pady=5)
            ttk.Button(btn_frame, text="Close", command=self.destroy).pack()
    
    
    class TribesTerrainExporterApp:
        """Main GUI application for terrain export.
        
        Provides an interactive interface for loading .ted terrain files,
        selecting DML files, exporting terrain data, and previewing results.
        
        Attributes:
            root: Main tkinter window
            terrain_obj: Loaded terrain object
            terrain_path: Path to loaded .ted file
            export_results: Generated export images and data
        """
        
        def __init__(self, root: tk.Tk) -> None:
            """Initialize the application GUI.
            
            Args:
                root: Main tkinter window
            """
            self.root = root
            self.root.title("TribesTerrainExporter - Terrain Export Tool")
            self.root.geometry("660x950")
            
            # Set window icon (mountain)
            self._set_window_icon()
            
            # Data
            self.terrain_path: Optional[Path] = None
            self.export_results: Optional[ExportResults] = None
            
            # Preview images (4 total)
            # 0: Terrain, 1: Heightmap, 2: Materials, 3: Flags
            self.preview_images: List[Optional[Any]] = [None] * NUM_PREVIEWS
            
            # PhotoImages for canvases (must keep references)
            self.active_photo: Optional[Any] = None
            self.thumbnail_photos: List[Optional[Any]] = [None] * NUM_PREVIEWS
            
            # Track which preview is active
            self.active_preview_index = DEFAULT_PREVIEW_INDEX
            
            # Track if full export has been done (not just preview)
            self.full_export_done = False
            
            # UI Variables
            self.tile_size_var = tk.IntVar(value=Defaults.TILE_SIZE)
            
            # Setup UI
            self.setup_ui()
            self.update_button_states()
        
        def _set_window_icon(self) -> None:
            """Set the window icon to a mountain."""
            try:
                # Create a simple 32x32 mountain icon
                icon_size = 32
                icon = Image.new('RGBA', (icon_size, icon_size), (0, 0, 0, 0))
                
                # Draw a simple mountain shape (triangle with snow cap)
                from PIL import ImageDraw
                draw = ImageDraw.Draw(icon)
                
                # Mountain body (dark green/brown)
                mountain_color = (101, 67, 33, 255)  # Brown
                draw.polygon([
                    (16, 4),      # Peak
                    (2, 30),      # Bottom left
                    (30, 30),     # Bottom right
                ], fill=mountain_color)
                
                # Snow cap (white)
                snow_color = (255, 255, 255, 255)
                draw.polygon([
                    (16, 4),      # Peak
                    (10, 14),     # Left edge of snow
                    (22, 14),     # Right edge of snow
                ], fill=snow_color)
                
                # Convert to PhotoImage and set as icon
                self._icon_photo = ImageTk.PhotoImage(icon)
                self.root.iconphoto(True, self._icon_photo)
                
            except Exception as e:
                # If icon creation fails, just continue without it
                logger.debug(f"Could not set window icon: {e}")
        
        def setup_ui(self) -> None:
            """Create and layout the user interface."""
            # Main container
            main_container = ttk.Frame(self.root, padding="10")
            main_container.pack(fill=tk.BOTH, expand=True)
            
            # Top section: Active preview and controls
            top_frame = ttk.Frame(main_container)
            top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False, pady=(0, 10))
            
            # Left: Active preview
            preview_frame = ttk.LabelFrame(top_frame, text="Preview", padding="5")
            preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
            
            self.active_canvas = Canvas(preview_frame, bg='#2b2b2b', cursor='hand2', width=400, height=300)
            self.active_canvas.pack(fill=tk.BOTH, expand=True)
            self.active_canvas.bind('<Button-1>', lambda e: self.show_active_full_size())
            
            # Right: Settings
            settings_frame = ttk.LabelFrame(top_frame, text="Export Settings", padding="5")
            settings_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
            settings_frame.config(width=320)
            settings_frame.pack_propagate(False)
            
            row = 0
            self.load_btn = ttk.Button(settings_frame, text="📁 Load .ted File...",
                                       command=self.load_ted_file, width=25)
            self.load_btn.grid(row=row, column=0, columnspan=3, pady=5)
            
            row += 1
            ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=row, column=0,
                                                                      columnspan=3, sticky=tk.EW, pady=10)
            
            row += 1
            ttk.Label(settings_frame, text="Tile Size:").grid(row=row, column=0, sticky=tk.W, pady=5)
            tile_combo = ttk.Combobox(settings_frame, textvariable=self.tile_size_var,
                                      values=VALID_TILE_SIZES,
                                      state='readonly', width=10)
            tile_combo.grid(row=row, column=1, sticky=tk.W, pady=5, padx=(5, 0))
            ttk.Label(settings_frame, text="pixels").grid(row=row, column=2, sticky=tk.W, pady=5, padx=(5, 0))
            
            row += 1
            ttk.Separator(settings_frame, orient=tk.HORIZONTAL).grid(row=row, column=0,
                                                                      columnspan=3, sticky=tk.EW, pady=10)
            
            row += 1
            self.export_btn = ttk.Button(settings_frame, text="🎨 Export Terrain",
                                         command=self.export_terrain, width=25)
            self.export_btn.grid(row=row, column=0, columnspan=3, pady=5)
            
            row += 1
            self.save_btn = ttk.Button(settings_frame, text="💾 Save All...",
                                       command=self.save_exports, width=25)
            self.save_btn.grid(row=row, column=0, columnspan=3, pady=5)
            
            # Middle section: Material info display
            info_frame = ttk.LabelFrame(main_container, text="Material Info", padding="5")
            info_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
            
            self.info_text = tk.Text(info_frame, height=6, wrap=tk.NONE, font=('Courier', 9))
            self.info_text.pack(fill=tk.X)
            self.info_text.config(state=tk.DISABLED)
            
            # Status bar (pack from bottom first so it doesn't get hidden)
            self.status_label = ttk.Label(main_container, text="Ready - Load a .ted file to begin",
                                          relief=tk.SUNKEN, anchor=tk.W)
            self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
            
            # Bottom section: Thumbnail previews
            thumbnail_frame = ttk.LabelFrame(main_container, text="Export Previews (Click to show above)", padding="5")
            thumbnail_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            thumbnail_grid = ttk.Frame(thumbnail_frame)
            thumbnail_grid.pack(fill=tk.BOTH, expand=True)
            
            # Configure grid for 4 columns
            for col in range(NUM_PREVIEWS):
                thumbnail_grid.grid_columnconfigure(col, weight=1)
            thumbnail_grid.grid_rowconfigure(0, weight=0)
            thumbnail_grid.grid_rowconfigure(1, weight=1)
            
            # Create 4 thumbnail previews
            self.thumbnail_canvases: List[Canvas] = []
            self.thumbnail_labels: List[ttk.Label] = []
            
            for i in range(NUM_PREVIEWS):
                # Label
                label = ttk.Label(thumbnail_grid, text=PREVIEW_LABELS[i], font=('TkDefaultFont', 8, 'bold'))
                label.grid(row=0, column=i, pady=(0, 5))
                self.thumbnail_labels.append(label)
                
                # Canvas - stretch to fill available space
                canvas = Canvas(thumbnail_grid, width=LAYER_PREVIEW_SIZE,
                               height=LAYER_PREVIEW_SIZE, bg='#2b2b2b', cursor='hand2')
                canvas.grid(row=1, column=i, padx=2, sticky='nsew')
                canvas.bind('<Button-1>', lambda e, idx=i: self.on_thumbnail_click(idx))
                self.thumbnail_canvases.append(canvas)
        
        def load_ted_file(self) -> None:
            """Load .ted terrain file and immediately generate previews."""
            filetypes = [
                ("TED files", "*.ted"),
                ("All files", "*.*")
            ]
            
            filename = filedialog.askopenfilename(
                title="Select Terrain File",
                filetypes=filetypes
            )
            
            if not filename:
                return
            
            try:
                self.terrain_path = Path(filename)
                logger.info("✓ Selected terrain file: %s", self.terrain_path.name)
                
                # Update status to show we're loading
                self.status_label.config(text=f"Loading {self.terrain_path.name}...")
                self.root.update_idletasks()
                
                # Immediately run export to generate previews (use 16px for fast preview)
                self._load_and_export(preview_mode=True)
                
            except (IOError, OSError, ValueError) as e:
                messagebox.showerror(MSG_ERROR_TITLE,
                    f"Failed to load {filename}:\n{str(e)}")
                self.update_button_states()
        
        def _load_and_export(self, preview_mode: bool = False) -> None:
            """Load terrain and generate exports immediately.
            
            Args:
                preview_mode: If True, uses small tile size (16px) for fast preview generation.
                             If False, uses the tile size selected in the dropdown.
            """
            if self.terrain_path is None:
                return
            
            # Disable buttons during export
            self.load_btn.config(state='disabled')
            self.export_btn.config(state='disabled')
            self.save_btn.config(state='disabled')
            
            # Log appropriate message based on mode
            if preview_mode:
                logger.info(f"🔍 Generating preview for {self.terrain_path.name}")
            else:
                logger.info(MSG_GENERATING.format(self.terrain_path.name))
            
            try:
                # Create config - use 16px tile size for fast preview, full size for export
                tile_size = 16 if preview_mode else self.tile_size_var.get()
                config = ExportConfig(
                    tile_size=tile_size,
                    debug=False
                )
                
                # Progress callback
                def progress_callback(percent: float, message: str) -> None:
                    self.status_label.config(text=f"{message} ({percent:.1f}%)")
                    self.root.update_idletasks()
                
                # Export to memory
                exporter = TerrainExporter(config)
                self.export_results = exporter.export_to_memory(
                    self.terrain_path,
                    progress_callback
                )
                
                if self.export_results is None:
                    messagebox.showerror(MSG_ERROR_TITLE, "Export failed. Check console for details.")
                    self.update_button_states()
                    return
                
                # Update preview images
                self.preview_images[0] = self.export_results.terrain_texture
                self.preview_images[1] = self.export_results.heightmap_vis
                self.preview_images[2] = self.export_results.materials_vis
                self.preview_images[3] = self.export_results.flags_vis
                
                # Update all thumbnails
                for i in range(NUM_PREVIEWS):
                    self.update_thumbnail(i)
                
                # Update active preview (default to terrain texture)
                self.active_preview_index = 0
                self.update_active_preview()
                
                # Update info display with stats
                self.update_info_display(self.export_results.stats)
                
                # Track if this was a full export (not just preview)
                if not preview_mode:
                    self.full_export_done = True
                    self.status_label.config(text=MSG_SUCCESS_EXPORT)
                    logger.info(MSG_SUCCESS_EXPORT)
                else:
                    self.status_label.config(text="✓ Preview generated")
                    logger.info("✓ Preview generated successfully")
                
                self.update_button_states()
                
            except (ValueError, RuntimeError, MemoryError) as e:
                logger.error("✗ Error generating terrain: %s", e)
                messagebox.showerror(MSG_ERROR_TITLE, f"Failed to generate terrain:\n{str(e)}")
                self.update_button_states()
        
        def update_info_display(self, stats: Optional[Dict[str, Any]] = None) -> None:
            """Update material info text display."""
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete('1.0', tk.END)
            
            if self.terrain_path:
                text = f"Terrain File: {self.terrain_path.name}\n"
                if stats:
                    text += f"Name: {stats.get('terrain_name', 'Unknown')}\n"
                    text += f"DML: {stats.get('dml_name', 'Unknown')}\n"
                    text += f"Size: {stats.get('size_x', '?')}×{stats.get('size_y', '?')} quads\n"
                    text += f"Heightmap: {stats.get('heightmap_size', '?')} pixels\n"
                    text += f"Materials: {stats.get('material_count', '?')} defined\n"
                else:
                    text += "Load and export to see terrain details.\n"
                self.info_text.insert(tk.END, text)
            else:
                self.info_text.insert(tk.END, "No terrain file loaded.\n")
            
            self.info_text.config(state=tk.DISABLED)
        
        def update_active_preview(self) -> None:
            """Update the active preview canvas with selected image."""
            image = self.preview_images[self.active_preview_index]
            if image is None:
                return
            
            # Get canvas size
            self.active_canvas.update_idletasks()
            canvas_width = self.active_canvas.winfo_width()
            canvas_height = self.active_canvas.winfo_height()
            
            if canvas_width <= 1:
                canvas_width = 400
            if canvas_height <= 1:
                canvas_height = 300
            
            # Scale to fit within canvas (maintain aspect ratio)
            img_width, img_height = image.size
            scale = min(canvas_width / img_width, canvas_height / img_height)
            preview_width = int(img_width * scale)
            preview_height = int(img_height * scale)
            
            # Use BILINEAR for speed
            preview = image.resize(
                (preview_width, preview_height),
                Image.Resampling.BILINEAR
            )
            
            # Convert to RGB if grayscale for display
            if preview.mode == 'L':
                preview = preview.convert('RGB')
            
            self.active_photo = ImageTk.PhotoImage(preview)
            
            # Clear and display centered
            self.active_canvas.delete('all')
            x = (canvas_width - preview_width) // 2
            y = (canvas_height - preview_height) // 2
            self.active_canvas.create_image(x, y, anchor=tk.NW, image=self.active_photo)
        
        def update_thumbnail(self, index: int) -> None:
            """Update a single thumbnail preview."""
            image = self.preview_images[index]
            if image is None:
                return
            
            canvas = self.thumbnail_canvases[index]
            
            # Get canvas actual size
            canvas.update_idletasks()
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            if canvas_width <= 1:
                canvas_width = 100
            if canvas_height <= 1:
                canvas_height = 100
            
            # Scale to fill height (shows vertical slice, centered horizontally)
            img_width, img_height = image.size
            scale = canvas_height / img_height
            preview_width = int(img_width * scale)
            preview_height = int(img_height * scale)
            
            # Resize
            preview = image.resize(
                (preview_width, preview_height),
                Image.Resampling.NEAREST
            )
            
            # Convert to RGB if grayscale
            if preview.mode == 'L':
                preview = preview.convert('RGB')
            
            photo = ImageTk.PhotoImage(preview)
            self.thumbnail_photos[index] = photo  # Keep reference
            
            # Clear and display centered horizontally
            canvas.delete('all')
            x = (canvas_width - preview_width) // 2
            y = 0  # Align to top since we're filling height
            canvas.create_image(x, y, anchor=tk.NW, image=photo)
        
        def on_thumbnail_click(self, index: int) -> None:
            """Handle click on a thumbnail - show in active preview."""
            if self.preview_images[index] is not None:
                self.active_preview_index = index
                self.update_active_preview()
        
        def show_active_full_size(self) -> None:
            """Open full-size viewer for active preview."""
            image = self.preview_images[self.active_preview_index]
            if image is None:
                return
            
            # Convert grayscale to RGB for viewer
            if image.mode == 'L':
                image = image.convert('RGB')
            
            title = f"{PREVIEW_LABELS[self.active_preview_index]} - Full Size"
            FullSizeViewer(self.root, image, title)
        
        def export_terrain(self) -> None:
            """Re-export terrain with current settings (tile size, debug mode)."""
            if self.terrain_path is None:
                messagebox.showwarning("No Terrain", MSG_NO_TED)
                return
            
            # Use shared export logic
            self._load_and_export()
        
        def save_exports(self) -> None:
            """Save all exported files to selected directory."""
            if self.export_results is None:
                messagebox.showwarning("No Exports", MSG_NO_EXPORTS)
                return
            
            directory = filedialog.askdirectory(
                title="Select Directory to Save Exports"
            )
            
            if not directory:
                return
            
            try:
                output_dir = Path(directory)
                base_name = self.terrain_path.stem if self.terrain_path else "terrain"
                
                # Save terrain texture
                if self.export_results.terrain_texture:
                    path = output_dir / f"{base_name}_terrain_texture.png"
                    self.export_results.terrain_texture.save(str(path), 'PNG')
                    logger.info(MSG_SUCCESS_SAVED, path.name,
                              self.export_results.terrain_texture.size[0],
                              self.export_results.terrain_texture.size[1])
                
                # Save heightmap PNG (16-bit)
                if self.export_results.heightmap_16bit is not None:
                    path = output_dir / f"{base_name}_heightmap.png"
                    ImageExporter.save_16bit_png(
                        self.export_results.heightmap_16bit, path, "Heightmap"
                    )
                
                # Save heightmap RAW
                if self.export_results.heightmap_raw:
                    path = output_dir / f"{base_name}_heightmap.raw"
                    path.write_bytes(self.export_results.heightmap_raw)
                    logger.info("✓ Saved: %s", path.name)
                
                # Save heightmap visualization
                if self.export_results.heightmap_vis:
                    path = output_dir / f"{base_name}_heightmap_vis.png"
                    self.export_results.heightmap_vis.save(str(path), 'PNG')
                    logger.info("✓ Saved: %s", path.name)
                
                # Save materials
                if self.export_results.materials_raw is not None:
                    path = output_dir / f"{base_name}_materials_raw.bmp"
                    ImageExporter.save_8bit_bmp(
                        self.export_results.materials_raw, path, "Materials (raw)"
                    )
                
                if self.export_results.materials_vis:
                    path = output_dir / f"{base_name}_materials_vis.bmp"
                    self.export_results.materials_vis.save(str(path), 'BMP')
                    logger.info("✓ Saved: %s", path.name)
                
                # Save flags
                if self.export_results.flags_raw is not None:
                    path = output_dir / f"{base_name}_material_flags_raw.bmp"
                    ImageExporter.save_8bit_bmp(
                        self.export_results.flags_raw, path, "Flags (raw)"
                    )
                
                if self.export_results.flags_vis:
                    path = output_dir / f"{base_name}_material_flags_vis.bmp"
                    self.export_results.flags_vis.save(str(path), 'BMP')
                    logger.info("✓ Saved: %s", path.name)
                
                # Save metadata
                if self.export_results.metadata_text:
                    path = output_dir / f"{base_name}_meta.txt"
                    path.write_text(self.export_results.metadata_text)
                    logger.info("✓ Saved: %s", path.name)
                
                messagebox.showinfo(MSG_SUCCESS_TITLE, f"Exports saved to:\n{output_dir}")
                
            except (IOError, OSError, PermissionError) as e:
                logger.error("✗ Error saving exports: %s", e)
                messagebox.showerror(MSG_ERROR_TITLE, f"Failed to save exports:\n{str(e)}")
        
        def update_button_states(self) -> None:
            """Enable/disable buttons based on application state."""
            has_terrain = self.terrain_path is not None
            has_full_export = self.full_export_done and self.export_results is not None
            
            state_export = 'normal' if has_terrain else 'disabled'
            state_save = 'normal' if has_full_export else 'disabled'
            
            self.export_btn.config(state=state_export)
            self.save_btn.config(state=state_save)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main_cli() -> None:
    """Command-line interface for terrain export.
    
    Handles argument parsing and routes to either GUI or CLI mode.
    """
    parser = argparse.ArgumentParser(
        description='Export Starsiege: Tribes .ted terrain files to heightmaps and textures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # GUI Mode (no arguments)
  python TribesTerrainExporter.py
  
  # CLI Mode
  python TribesTerrainExporter.py myterrain.ted
  python TribesTerrainExporter.py myterrain.ted -o ./exports
  python TribesTerrainExporter.py myterrain.ted --tile-size 128 --debug
        '''
    )
    
    parser.add_argument('terrain_file', nargs='?', help='Path to .ted terrain file')
    parser.add_argument(
        '-o', '--output',
        dest='output_dir',
        help='Output directory (default: ./<terrain_name>_export)'
    )
    parser.add_argument(
        '-t', '--tile-size',
        dest='tile_size',
        type=int,
        default=Defaults.TILE_SIZE,
        help=f'Size of each texture tile in pixels (default: {Defaults.TILE_SIZE})'
    )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Enable debug mode (draw borders and flag values on tiles)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose (debug) logging'
    )
    
    args = parser.parse_args()
    
    # If no terrain file, launch GUI
    if args.terrain_file is None:
        if not TKINTER_AVAILABLE:
            logger.error("ERROR: tkinter is not available. Cannot launch GUI mode.")
            logger.error("Please provide a terrain file for CLI mode.")
            sys.exit(1)
        
        logger.info("Launching GUI mode...")
        root = tk.Tk()
        app = TribesTerrainExporterApp(root)
        root.mainloop()
        return
    
    # CLI mode
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    terrain_path = Path(args.terrain_file)
    if not terrain_path.exists():
        logger.error(f"File not found: {args.terrain_file}")
        sys.exit(1)
    
    logger.info(f"Processing: {args.terrain_file}")
    if args.debug:
        logger.info("Debug mode: ON")
    logger.info(f"Tile size: {args.tile_size}x{args.tile_size} pixels")
    
    success = export_terrain(
        args.terrain_file,
        args.output_dir,
        args.tile_size,
        args.debug
    )
    
    sys.exit(0 if success else 1)


def main() -> int:
    """Main entry point for the application.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Run the CLI (which may launch GUI)
        main_cli()
        return 0
    except KeyboardInterrupt:
        logger.info("\n\nOperation cancelled by user.")
        return 1
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
