# Polygon Set Operations - Technical Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Architecture](#architecture)
4. [Module Structure](#module-structure)
5. [Core Functions](#core-functions)
6. [Set Operations Algorithms](#set-operations-algorithms)
7. [Usage Guide](#usage-guide)
8. [File Format Specifications](#file-format-specifications)
9. [Error Handling](#error-handling)
10. [Examples](#examples)

---

## Overview

**PolyWithText.py** is a Python application that provides a comprehensive system for managing polygons and performing geometric set operations (union, intersection, and difference) on them. The application uses computational geometry algorithms to compute these operations using only the standard Python libraries `turtle` and `math`.

### Key Features
- **Polygon Management**: Add, save, display, and remove polygons
- **Set Operations**: Union, Intersection, and Difference operations
- **Visualization**: Graphical display using Turtle graphics
- **File I/O**: Import/export polygons from text files
- **Interactive Menu**: User-friendly command-line interface

---

## System Requirements

### Software Requirements
- **Python Version**: 3.6 or higher
- **Required Libraries**:
  - `turtle` (standard library)
  - `math` (standard library)
- **Operating System**: Cross-platform (Windows, Linux, macOS)

### Hardware Requirements
- Minimal requirements (standard Python installation)
- Display capable of running Turtle graphics

---

## Architecture

### Design Pattern
The application follows a **modular design** with clear separation of concerns:

1. **Polygon Utilities Module**: I/O operations and display functions
2. **Geometric Algorithms Module**: Core computational geometry functions
3. **Set Operations Module**: Union, Intersection, and Difference implementations
4. **Main Application Module**: User interface and program flow

### Data Structures
- **Polygon Representation**: List of `(x, y)` tuples
- **Polygon Storage**: Dictionary `{name: [(x, y), ...]}`
- **Graph Structure**: Dictionary for edge connectivity `{point_key: [neighbor_keys]}`

---

## Module Structure

### 1. Polygon Utilities

#### `read_polygon_from_input()`
- **Purpose**: Read polygon vertices from keyboard input
- **Returns**: `list` of `(x, y)` tuples
- **User Interaction**: Prompts for number of vertices and coordinates

#### `read_polygon_from_file(filename)`
- **Purpose**: Read polygon vertices from a text file
- **Parameters**: 
  - `filename` (str): Path to input file
- **Returns**: `list` of `(x, y)` tuples
- **File Format**: Each line contains "x y" (space-separated)

#### `write_polygon_to_file(points, filename)`
- **Purpose**: Save polygon vertices to a text file
- **Parameters**:
  - `points` (list): List of `(x, y)` tuples
  - `filename` (str): Output file path
- **File Format**: Each vertex written as "x y\n"

#### `display_polygons(polygons, labels, highlight_index=None)`
- **Purpose**: Visualize polygons using Turtle graphics
- **Parameters**:
  - `polygons` (list): List of polygon vertex lists
  - `labels` (list): List of polygon names
  - `highlight_index` (int, optional): Index of polygon to highlight
- **Features**:
  - Color-coded polygons
  - Thicker lines for highlighted polygons (pensize 3)
  - Labels displayed at first vertex

### 2. Geometric Algorithms

#### `point_on_segment(p, a, b, epsilon=1e-9)`
- **Purpose**: Check if point lies on line segment
- **Algorithm**: Cross product for collinearity + bounding box check
- **Returns**: `bool`

#### `line_intersection(p1, p2, p3, p4)`
- **Purpose**: Find intersection point of two line segments
- **Algorithm**: Parametric line equations
- **Returns**: `(x, y)` tuple or `None`

#### `point_in_polygon(point, polygon)`
- **Purpose**: Determine if point is inside polygon
- **Algorithm**: Ray casting algorithm
- **Returns**: `bool`

#### `get_all_intersections(poly_a, poly_b)`
- **Purpose**: Find all intersection points between two polygons
- **Algorithm**: Check every edge pair
- **Returns**: List of intersection tuples with metadata

#### `distance(p1, p2)`
- **Purpose**: Calculate Euclidean distance
- **Returns**: `float`

#### `cross_product(o, a, b)`
- **Purpose**: Calculate cross product for orientation
- **Returns**: `float` (positive = counter-clockwise, negative = clockwise)

#### `convex_hull_union(poly_a, poly_b)`
- **Purpose**: Compute convex hull of disjoint polygons
- **Algorithm**: Graham scan algorithm
- **Returns**: Convex hull as list of `(x, y)` tuples

### 3. Set Operations

#### `union(poly_a, poly_b)`
- **Purpose**: Compute union of two polygons
- **Algorithm**: Boundary tracing following outside edges
- **Returns**: Result polygon as list of `(x, y)` tuples

#### `intersection(poly_a, poly_b)`
- **Purpose**: Compute intersection of two polygons
- **Algorithm**: Boundary tracing following inside edges
- **Returns**: Result polygon or empty list if no overlap

#### `difference(poly_a, poly_b)`
- **Purpose**: Compute A - B (parts of A not in B)
- **Algorithm**: Boundary tracing with hole handling
- **Returns**: Result polygon or empty list if A is inside B

---

## Set Operations Algorithms

### Union Algorithm

**Objective**: Combine two polygons into one, including all points in either polygon.

**Steps**:
1. Find all intersection points between polygon edges
2. Create enhanced vertex lists with intersections inserted
3. Build connectivity graph of edges that are **outside** the other polygon
4. Trace boundary following outside edges in counter-clockwise order
5. Return the traced boundary as result polygon

**Edge Classification**:
- Edge is included if its midpoint is **outside** the other polygon

### Intersection Algorithm

**Objective**: Find overlapping region between two polygons.

**Steps**:
1. Find all intersection points between polygon edges
2. Create enhanced vertex lists with intersections inserted
3. Build connectivity graph of edges that are **inside** the other polygon
4. Trace boundary following inside edges in clockwise order
5. Return the traced boundary as result polygon

**Edge Classification**:
- Edge is included if its midpoint is **inside** the other polygon

### Difference Algorithm

**Objective**: Find parts of polygon A that are not in polygon B.

**Steps**:
1. Find all intersection points between polygon edges
2. Create enhanced vertex lists with intersections inserted
3. Build connectivity graph:
   - Edges from A that are outside B (part of result)
   - Edges from B that are inside A (traced in reverse for holes)
4. Trace boundary following outside edges of A
5. Return the traced boundary as result polygon

**Edge Classification**:
- Include edges from A if midpoint is outside B
- Include edges from B (reversed) if midpoint is inside A (for holes)

### Boundary Tracing Algorithm

**Common to all operations**:
1. Start at leftmost point (or appropriate starting point)
2. For each vertex:
   - Calculate incoming direction
   - Find next vertex with smallest appropriate turn angle
   - Add vertex to result
   - Mark edge as visited
3. Continue until returning to start point
4. Clean result (remove duplicates, ensure closure)

---

## Usage Guide

### Starting the Application

```bash
python PolyWithText.py
```

### Menu Options

#### 1. Add Polygon
- **Keyboard Input**: Enter 'k', then provide number of vertices and coordinates
- **File Input**: Enter 'f', then provide filename
- **Example**:
  ```
  Enter polygon name: P1
  Enter 'k' for keyboard, 'f' for file: f
  Filename: P1.txt
  ```

#### 2. Save Polygon to File
- Enter polygon name and output filename
- Polygon vertices are saved in "x y" format

#### 3. Show Polygons
- Displays all polygons in Turtle graphics window
- Each polygon shown in different color with label

#### 4. Combine (Union/Intersection/Difference)
- **Union**: Enter 'union' or 'u'
  - Prompts for two polygon names
  - Returns combined polygon
- **Intersection**: Enter 'intersection' or 'i'
  - Prompts for two polygon names
  - Returns overlapping region
- **Difference**: Enter 'difference' or 'd'
  - Prompts: "Difference from ?" and "Difference with ?"
  - Returns A - B (parts of first polygon not in second)

#### 5. Remove Polygon
- Lists available polygons
- Enter name of polygon to remove

#### 6. Quit
- Exits the application

---

## File Format Specifications

### Input File Format

**Structure**: Plain text file with one vertex per line

**Format**: 
```
x1 y1
x2 y2
x3 y3
...
```

**Example** (`P1.txt`):
```
0 0
0 80
80 80
80 0
```

**Rules**:
- Each line contains two space-separated numbers (x coordinate, y coordinate)
- Empty lines are automatically skipped
- No header or footer required
- Coordinates can be integers or floating-point numbers

### Output File Format

**Structure**: Same as input format

**Format**:
```
x1 y1
x2 y2
x3 y3
...
```

**Note**: The last vertex may be duplicated (first vertex repeated) to indicate closure.

---

## Error Handling

### Input Validation
- **Missing Polygons**: Error message if polygon name not found
- **Empty Files**: Handled gracefully (empty polygon list)
- **Invalid Coordinates**: Python will raise ValueError for non-numeric input

### Operation Errors
- **Union Failure**: Returns error message, falls back to convex hull if possible
- **Intersection Failure**: Returns empty list if polygons don't overlap
- **Difference Failure**: Returns empty list if A is completely inside B

### Edge Cases Handled
- Empty polygons
- Disjoint polygons
- One polygon inside another
- No intersections
- Invalid polygon names

---

## Examples

### Example 1: Basic Union Operation

**Input Polygons**:
- P1: Square from (0,0) to (80,80)
- P2: Square from (40,40) to (120,120)

**Operation**: Union

**Result**: L-shaped polygon combining both squares

### Example 2: Intersection Operation

**Input Polygons**:
- P1: Square from (0,0) to (80,80)
- P2: Square from (40,40) to (120,120)

**Operation**: Intersection

**Result**: Square from (40,40) to (80,80) - the overlapping region

### Example 3: Difference Operation

**Input Polygons**:
- P1: Square from (0,0) to (80,80)
- P2: Square from (40,40) to (120,120)

**Operation**: P1 - P2

**Result**: L-shaped polygon (P1 minus the overlapping region)

**Operation**: P2 - P1

**Result**: L-shaped polygon (P2 minus the overlapping region)

### Example Workflow

```
1. Add polygon (keyboard/file): 1
   Enter polygon name: P1
   Enter 'k' for keyboard, 'f' for file: f
   Filename: P1.txt

2. Add polygon (keyboard/file): 1
   Enter polygon name: P2
   Enter 'k' for keyboard, 'f' for file: f
   Filename: P2.txt

3. Show polygons: 3
   [Displays P1 and P2 in Turtle window]

4. Combine (union/intersection/difference): 4
   Enter operation: union
   Name first polygon: P1
   Name second polygon: P2
   Result polygon name: P_union
   [Displays all three polygons with P_union highlighted]

5. Quit: 6
```

---

## Version Information

**Version**: 1.0  
**Last Updated**: 2025  
**Author**: Based on MA1008 Mini Project requirements  
**License**: Educational/Project Use

---

## References

- Computational Geometry Algorithms
- Ray Casting Algorithm for Point-in-Polygon
- Graham Scan Algorithm for Convex Hull
- Weiler-Atherton Algorithm (inspired approach for set operations)

---

## Appendix: Function Reference

### Quick Reference Table

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `read_polygon_from_input()` | Read from keyboard | None | `list[(x,y)]` |
| `read_polygon_from_file()` | Read from file | `str` | `list[(x,y)]` |
| `write_polygon_to_file()` | Save to file | `list[(x,y)], str` | None |
| `display_polygons()` | Visualize | `list, list, int?` | None |
| `union()` | Union operation | `list[(x,y)], list[(x,y)]` | `list[(x,y)]` |
| `intersection()` | Intersection operation | `list[(x,y)], list[(x,y)]` | `list[(x,y)]` |
| `difference()` | Difference operation | `list[(x,y)], list[(x,y)]` | `list[(x,y)]` |
| `point_in_polygon()` | Point test | `(x,y), list[(x,y)]` | `bool` |
| `line_intersection()` | Segment intersection | 4 points | `(x,y)?` |

---

*End of Technical Documentation*

