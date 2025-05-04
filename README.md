# ShanShui.py

A Python-based SVG Chinese landscape generator inspired by traditional Shan Shui paintings. This project is a Python implementation of [Lingdong Huang's Shan-Shui-Inf](https://github.com/LingDong-/shan-shui-inf) JavaScript project.

## What is Shan Shui?

"Shan Shui" (山水) literally translates to "mountain-water" in Chinese. It refers to a traditional style of Chinese landscape painting that has existed since the 5th century. These paintings typically depict mountains, rivers, and often small human figures, capturing the essence of nature rather than creating a photorealistic depiction.

This generator creates procedurally-generated vector-format Chinese landscapes in SVG format using noise functions and mathematical algorithms to model mountains, trees, and other natural elements.

## Installation

ShanShui.py has no external dependencies and works with standard Python 3.6+.

1. Clone or download this repository:
   ```
   git clone https://github.com/MushroomFleet/SVG_ShanShui-cli
   cd shangshui
   ```

2. Verify the installation:
   ```
   python ShanShui.py --width 800 --height 600
   ```

   This will generate a landscape SVG file with the current timestamp in your current directory.

## Usage

Basic command syntax:

```
python ShanShui.py [--width WIDTH] [--height HEIGHT] [--seed SEED] [--output FILENAME]
```

### Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--width` | Width of the output SVG in pixels | 3000 |
| `--height` | Height of the output SVG in pixels | 800 |
| `--seed` | Random seed for reproducible generation (optional) | Current timestamp |
| `--output` | Output file path (optional) | `shanshui_TIMESTAMP.svg` |

## Step-by-Step Instructions

### Generating Your First Landscape

1. Open a terminal or command prompt
2. Navigate to the directory containing `ShanShui.py`
3. Run the basic command:
   ```
   python ShanShui.py
   ```
4. Find the generated SVG file in the same directory (named with the current timestamp)
5. Open the SVG file in a web browser or image viewer

### Customizing Dimensions

Specify the width and height to create landscapes of different sizes:

```
python ShanShui.py --width 1920 --height 1080
```

This is useful for creating landscapes that fit specific aspect ratios or screen sizes.

### Using Seeds for Reproducibility

To generate the same landscape multiple times or share a particular landscape:

1. Generate a landscape with a specific seed:
   ```
   python ShanShui.py --seed "my_favorite_mountain"
   ```

2. Share the seed value with others, who can then generate the exact same landscape:
   ```
   python ShanShui.py --seed "my_favorite_mountain"
   ```

### Saving with Custom Filenames

To save the output with a specific filename:

```
python ShanShui.py --output my_landscape.svg
```

This is helpful when organizing multiple generated landscapes.

## Examples

### Basic Generation

```
python ShanShui.py
```

Generates a standard landscape with default dimensions (3000×800) and saves it with a timestamp filename.

### High-Resolution Landscape

```
python ShanShui.py --width 3840 --height 2160 --output 4k_landscape.svg
```

Creates a 4K resolution landscape suitable for high-resolution displays or printing.

### Wide Panoramic View

```
python ShanShui.py --width 5000 --height 1000 --output panorama.svg
```

Creates a wide panoramic landscape with a 5:1 aspect ratio.

### Mobile Wallpaper

```
python ShanShui.py --width 1080 --height 1920 --output mobile_wallpaper.svg
```

Creates a portrait-oriented landscape suitable for mobile device wallpapers.

### Reproducible Generation

```
python ShanShui.py --seed "mountain123" --output mountain123.svg
```

Creates a landscape that will be identical every time this seed is used, allowing for reproducible results.

## How It Works

ShanShui.py generates landscapes using several key components:

1. **Custom PRNG (Pseudo-Random Number Generator)** - For consistent and reproducible random number generation
2. **Perlin Noise** - To create natural-looking variations in terrain
3. **Bezier Curves** - For smooth, natural shapes in trees and mountains
4. **Polygon Triangulation** - To efficiently generate complex mountain shapes
5. **SVG Generation** - To output the image in a scalable vector format

The generator creates different elements like distant mountains, middle-ground mountains, and foreground elements like trees, combining them into a coherent landscape composition.

## Acknowledgments

This project is a Python implementation based on the original [Shan-Shui-Inf](https://github.com/LingDong-/shan-shui-inf) by Lingdong Huang, which is a procedurally-generated vector-format infinitely-scrolling Chinese landscape for the browser.
