# TribesTerrainExporter 🏔️

> **Starsiege: Tribes Terrain File Exporter**

Export Starsiege: Tribes `.ted` terrain files to heightmaps, material maps, and rendered textures - all from a standalone Python script with automatic dependency management.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

## ✨ Features

- 🏔️ **Complete Terrain Export** - Heightmaps, material maps, and fully rendered terrain textures
- 🚀 **Automatic Dependency Management** - Installs NumPy, Pillow, and TribesToBlender on first run
- 📦 **Self-Contained** - Downloads required TribesToBlender modules automatically from GitHub
- 🗜️ **Archive Handling** - Extracts textures from DML.zip and terrain data from Terrain.zip
- 🎨 **DML Parsing** - Full material definition support with texture mapping
- 🔄 **Tile Transformations** - Proper rotation and flip handling based on material flags
- 🐛 **Debug Mode** - Visual debugging with tile borders and flag annotations

## 📋 Requirements

- **Python 3.7+**
- **Dependencies** (auto-installed on first run):
  - NumPy
  - Pillow

## 🚀 Quick Start

### Installation

1. Clone or download the repository:
```bash
git clone https://github.com/yourusername/TribesTerrainExporter.git
cd TribesTerrainExporter
```

2. Run the script (dependencies install automatically):
```bash
python TribesTerrainExporter.py Raindance.ted
```

### Basic Usage

Export a terrain file with default settings (64x64 pixel tiles):

```bash
python TribesTerrainExporter.py Raindance.ted
```

This creates a `Raindance_export/` folder with all exported files.

### Custom Output Directory

Specify where to save exported files:

```bash
python TribesTerrainExporter.py Raindance.ted -o ./my_exports
```

### Custom Tile Size

Change the texture tile size (default is 64px):

```bash
python TribesTerrainExporter.py Raindance.ted --tile-size 128
```

- **Smaller tiles** = Smaller output images (lower detail)
- **Larger tiles** = Larger output images (higher detail)

### Debug Mode

Enable debug mode to visualize tile transformations:

```bash
python TribesTerrainExporter.py Raindance.ted --debug
```

### Combined Options

```bash
python TribesTerrainExporter.py Raindance.ted -o ./exports --tile-size 32 -d
```

**CLI Options:**
- `-o, --output` - Output directory (default: `<terrain_name>_export`)
- `-t, --tile-size` - Tile size in pixels (default: 64)
- `-d, --debug` - Draw borders and flag values on tiles

## 📁 Directory Structure

```
TribesTerrainExporter/
├── TribesTerrainExporter.py    # Main script
├── Raindance.ted               # Example terrain file
├── dml/                        # DML material definitions & archives
│   ├── lush.dml                # Extracted DML files
│   ├── desert.dml
│   ├── *DML.zip                # Optional: DML archives (for texture extraction)
│   └── ...
├── textures/                   # Texture PNG files (extracted from DML.zip)
│   └── *.png
└── terrain_archives/           # Optional: *Terrain.zip files (for DML extraction)
    └── *.zip
```

## 📸 Output

The tool generates Unity-ready terrain data in `<terrain_name>_export/`:

### Heightmap Files
- **`<terrain>_heightmap.png`** - 16-bit PNG heightmap (0-65535 range)
- **`<terrain>_heightmap.raw`** - Raw 16-bit little-endian (Unity-compatible)

### Material Maps
- **`<terrain>_materials_raw.bmp`** - Raw material indices
- **`<terrain>_materials_vis.bmp`** - Visualization (scaled for visibility)

### Material Flags
- **`<terrain>_material_flags_raw.bmp`** - Raw flag values
- **`<terrain>_material_flags_vis.bmp`** - Visualization (scaled for visibility)

### Rendered Terrain
- **`<terrain>_terrain_texture.png`** - Fully assembled terrain texture

### Metadata
- **`<terrain>_meta.txt`** - Terrain metadata and statistics

## 🔧 How It Works

1. **Dependency Check** - Installs NumPy and Pillow if missing
2. **TribesToBlender Download** - Fetches required modules from GitHub (bovidi branch)
3. **Archive Extraction** - Extracts textures from `*DML.zip` and DML files from `*Terrain.zip`
4. **Terrain Parsing** - Reads `.ted` file (handles both plain and zipped formats)
5. **DML Parsing** - Loads material definitions and maps to texture files
6. **Data Export** - Generates heightmaps and material/flag maps
7. **Texture Rendering** - Assembles tiles with proper transformations (rotations, flips)

## 📊 Example: Raindance

```bash
python TribesTerrainExporter.py Raindance.ted
```

**Expected Output:**
- Terrain size: 256×256 squares
- Heightmap: 257×257 pixels
- With 64px tiles: 16384×16384 pixel terrain texture
- Materials: 184 total, ~104 unique used
- Flags: 31 unique values

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- **[TribesToBlender](https://github.com/tekrog/TribesToBlender)** - For the incredible reverse-engineering work and file format parsers
- Part of the **TribesToBlender** project for bringing classic Tribes game assets into modern engines

Shazbot! 🔥

---

## ⚠️ Disclaimer

This tool was **vibe coded** using **GitHub Copilot (Claude Opus 4.5)** by an author who freely admits to knowing **absolutely nothing about Python**.
