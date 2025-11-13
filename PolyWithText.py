import turtle
import math

# ---- Polygon Utilities ----

def read_polygon_from_input():
    """
    Read polygon vertices from keyboard input.
    
    Prompts the user to enter the number of vertices and then coordinates
    for each vertex. Returns a list of (x, y) tuples representing the polygon.
    
    Returns:
        list: List of (x, y) tuples representing polygon vertices
    """
    n = int(input("Enter number of vertices: "))
    points = []
    for i in range(n):
        x, y = map(float, input(f"Enter vertex {i+1} (x y): ").split())
        points.append((x, y))
    return points

def read_polygon_from_file(filename):
    """
    Read polygon vertices from a text file.
    
    Each line in the file should contain two space-separated numbers (x y).
    Empty lines are automatically skipped.
    
    Args:
        filename (str): Path to the file containing polygon vertices
        
    Returns:
        list: List of (x, y) tuples representing polygon vertices
    """
    with open(filename) as f:
        points = [tuple(map(float, line.split())) for line in f if line.strip()]
    return points

def write_polygon_to_file(points, filename):
    """
    Save polygon vertices to a text file.
    
    Writes each vertex as a line with format "x y" (space-separated coordinates).
    
    Args:
        points (list): List of (x, y) tuples representing polygon vertices
        filename (str): Path to the output file
    """
    with open(filename, 'w') as f:
        for x, y in points:
            f.write(f"{x} {y}\n")

def display_polygons(polygons, labels, highlight_index=None):
    """
    Display multiple polygons using Turtle graphics.
    
    Each polygon is drawn in a different color from a predefined color list.
    Optionally, one polygon can be highlighted with a thicker line width.
    
    Args:
        polygons (list): List of polygons, where each polygon is a list of (x, y) tuples
        labels (list): List of labels (strings) corresponding to each polygon
        highlight_index (int, optional): Index of polygon to highlight with thicker line.
                                        If None, all polygons use normal line width.
    """
    turtle.clearscreen()
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    
    # Draw each polygon
    for i, pts in enumerate(polygons):
        turtle.penup()
        turtle.pencolor(colors[i % len(colors)])
        
        # Set thicker line for highlighted polygon (pensize 3) vs normal (pensize 1)
        if highlight_index is not None and i == highlight_index:
            turtle.pensize(3)
        else:
            turtle.pensize(1)
        
        # Move to first vertex and start drawing
        turtle.goto(pts[0])
        turtle.pendown()
        
        # Draw edges to all subsequent vertices
        for pt in pts[1:]:
            turtle.goto(pt)
        
        # Close the polygon by returning to first vertex
        turtle.goto(pts[0])
        
        # Add label text at the first vertex
        turtle.penup()
        turtle.goto(pts[0])
        turtle.write(labels[i], font=('Arial', 12, 'bold'))
    
    turtle.hideturtle()

# ---- Set Operations (Basis for Pseudocode) ----

def point_on_segment(p, a, b, epsilon=1e-9):
    """
    Check if a point lies on a line segment.
    
    Uses cross product to check collinearity and bounding box to check
    if the point is within the segment's extent.
    
    Args:
        p (tuple): Point (x, y) to check
        a (tuple): First endpoint of segment (x, y)
        b (tuple): Second endpoint of segment (x, y)
        epsilon (float): Tolerance for floating point comparison
        
    Returns:
        bool: True if point p lies on segment ab, False otherwise
    """
    # Check if p is collinear with a and b using cross product
    cross = (b[1] - a[1]) * (p[0] - a[0]) - (b[0] - a[0]) * (p[1] - a[1])
    if abs(cross) > epsilon:
        return False
    
    # Check if p is within the bounding box of segment ab
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
            min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))

def line_intersection(p1, p2, p3, p4):
    """
    Find the intersection point of two line segments.
    
    Uses parametric line equations to find where segments (p1-p2) and (p3-p4) intersect.
    Returns the intersection point if both segments contain it, None otherwise.
    
    Args:
        p1 (tuple): First endpoint of first segment (x, y)
        p2 (tuple): Second endpoint of first segment (x, y)
        p3 (tuple): First endpoint of second segment (x, y)
        p4 (tuple): Second endpoint of second segment (x, y)
        
    Returns:
        tuple or None: Intersection point (x, y) if segments intersect, None otherwise
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # Calculate denominator for parametric equations
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None  # Lines are parallel (or coincident)
    
    # Calculate parameters t and u for the two line segments
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    # Check if intersection is within both segments (0 <= t, u <= 1)
    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y)
    return None

def point_in_polygon(point, polygon):
    """
    Determine if a point is inside a polygon using the ray casting algorithm.
    
    Casts a horizontal ray from the point to infinity and counts intersections
    with polygon edges. Odd number of intersections means point is inside.
    
    Args:
        point (tuple): Point (x, y) to test
        polygon (list): List of (x, y) tuples representing polygon vertices
        
    Returns:
        bool: True if point is inside polygon, False otherwise
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    # Check each edge of the polygon
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        
        # Check if ray crosses this edge
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    # Calculate x-coordinate of intersection
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside  # Toggle inside/outside state
        p1x, p1y = p2x, p2y
    
    return inside

def get_all_intersections(poly_a, poly_b):
    """
    Find all intersection points between edges of two polygons.
    
    Checks every edge of poly_a against every edge of poly_b to find
    all intersection points. Returns information about which edges intersect
    and at what points.
    
    Args:
        poly_a (list): First polygon as list of (x, y) tuples
        poly_b (list): Second polygon as list of (x, y) tuples
        
    Returns:
        list: List of tuples (intersection_point, edge_idx_a, edge_idx_b, 'a', 'b')
              Each intersection is stored twice (once for each polygon's perspective)
    """
    intersections = []
    n_a = len(poly_a)
    n_b = len(poly_b)
    
    # Check each edge of poly_a against each edge of poly_b
    for i in range(n_a):
        p1 = poly_a[i]
        p2 = poly_a[(i + 1) % n_a]
        for j in range(n_b):
            p3 = poly_b[j]
            p4 = poly_b[(j + 1) % n_b]
            inter = line_intersection(p1, p2, p3, p4)
            if inter:
                # Store intersection from both polygons' perspectives
                intersections.append((inter, i, j, 'a', 'b'))
                intersections.append((inter, j, i, 'b', 'a'))
    
    return intersections

def distance(p1, p2):
    """
    Calculate Euclidean distance between two points.
    
    Args:
        p1 (tuple): First point (x, y)
        p2 (tuple): Second point (x, y)
        
    Returns:
        float: Euclidean distance between p1 and p2
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_edges_with_intersections(poly, intersections, poly_id):
    """
    Get edges of polygon with intersection points inserted in order.
    
    For each edge of the polygon, finds all intersection points on that edge
    and inserts them in order along the edge (sorted by distance from start vertex).
    
    Args:
        poly (list): Polygon as list of (x, y) tuples
        intersections (list): List of intersection tuples from get_all_intersections
        poly_id (str): Identifier for this polygon ('a' or 'b')
        
    Returns:
        list: List of edges, where each edge is a list of points including
              start vertex, intersection points (in order), and end vertex
    """
    n = len(poly)
    edges = []
    
    for i in range(n):
        edge_points = [poly[i]]  # Start with the first vertex of the edge
        
        # Find all intersections on this edge
        edge_intersections = []
        for inter, idx_a, idx_b, id_a, id_b in intersections:
            if id_a == poly_id and idx_a == i:
                edge_intersections.append((inter, distance(poly[i], inter)))
        
        # Sort intersections by distance from start of edge
        edge_intersections.sort(key=lambda x: x[1])
        edge_points.extend([inter for inter, _ in edge_intersections])
        edge_points.append(poly[(i + 1) % n])  # Add end vertex
        edges.append(edge_points)
    
    return edges

def classify_edge(edge_start, edge_end, other_poly):
    """
    Classify whether an edge is inside or outside another polygon.
    
    Uses the midpoint of the edge to determine if the edge is inside
    the other polygon.
    
    Args:
        edge_start (tuple): Start point of edge (x, y)
        edge_end (tuple): End point of edge (x, y)
        other_poly (list): Other polygon as list of (x, y) tuples
        
    Returns:
        bool: True if edge is inside other_poly, False if outside
    """
    # Use midpoint of edge to determine classification
    mid_x = (edge_start[0] + edge_end[0]) / 2
    mid_y = (edge_start[1] + edge_end[1]) / 2
    midpoint = (mid_x, mid_y)
    return point_in_polygon(midpoint, other_poly)

def union(poly_a, poly_b):
    """
    Compute the union of two polygons using geometric algorithms.
    
    The union operation combines two polygons into one, including all points
    that are in either polygon. Uses a boundary tracing algorithm that follows
    the outer edges of both polygons.
    
    Algorithm:
    1. Find all intersection points between the two polygons
    2. Create enhanced vertex lists with intersections inserted
    3. Build a graph of edges that are outside the other polygon
    4. Trace the boundary following outside edges in counter-clockwise order
    
    Args:
        poly_a (list): First polygon as list of (x, y) tuples
        poly_b (list): Second polygon as list of (x, y) tuples
        
    Returns:
        list: Result polygon as list of (x, y) tuples representing the union
    """
    # Handle empty polygons
    if not poly_a:
        return poly_b
    if not poly_b:
        return poly_a
    
    # Find all intersection points between the two polygons
    intersections = get_all_intersections(poly_a, poly_b)
    
    # Handle special cases when polygons don't intersect
    if not intersections:
        # Check if one polygon completely contains the other
        if point_in_polygon(poly_a[0], poly_b):
            return poly_b  # poly_a is inside poly_b, so union is poly_b
        if point_in_polygon(poly_b[0], poly_a):
            return poly_a  # poly_b is inside poly_a, so union is poly_a
        # Polygons are disjoint - return convex hull of both
        return convex_hull_union(poly_a, poly_b)
    
    # Create enhanced vertex lists with intersection points inserted
    def create_enhanced_list(poly, intersections, poly_id):
        n = len(poly)
        enhanced = []
        
        for i in range(n):
            start = poly[i]
            end = poly[(i + 1) % n]
            
            # Find intersections on this edge
            edge_inters = []
            for inter, idx_a, idx_b, id_a, id_b in intersections:
                if id_a == poly_id and idx_a == i:
                    d = distance(start, inter)
                    edge_inters.append((inter, d))
            
            # Sort by distance
            edge_inters.sort(key=lambda x: x[1])
            
            # Add start point (only for first edge, or if different from last)
            if i == 0 or distance(start, enhanced[-1]) > 1e-6:
                enhanced.append(start)
            # Add intersections in order
            for inter, _ in edge_inters:
                if distance(inter, enhanced[-1]) > 1e-6:
                    enhanced.append(inter)
        
        return enhanced
    
    list_a = create_enhanced_list(poly_a, intersections, 'a')
    list_b = create_enhanced_list(poly_b, intersections, 'b')
    
    # Create point to index mapping
    def point_to_key(p):
        return (round(p[0], 8), round(p[1], 8))
    
    # Build connectivity graph for outside edges only
    graph = {}
    point_map = {}
    
    # Add all points to map
    for p in list_a + list_b:
        key = point_to_key(p)
        point_map[key] = p
    
    # Add edges from poly_a if outside poly_b
    n_a = len(list_a)
    for i in range(n_a):
        p1 = list_a[i]
        p2 = list_a[(i + 1) % n_a]
        k1 = point_to_key(p1)
        k2 = point_to_key(p2)
        
        # Check if edge is outside
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        if not point_in_polygon(mid, poly_b):
            if k1 not in graph:
                graph[k1] = []
            if k2 not in graph[k1]:
                graph[k1].append(k2)
    
    # Add edges from poly_b if outside poly_a
    n_b = len(list_b)
    for i in range(n_b):
        p1 = list_b[i]
        p2 = list_b[(i + 1) % n_b]
        k1 = point_to_key(p1)
        k2 = point_to_key(p2)
        
        # Check if edge is outside
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        if not point_in_polygon(mid, poly_a):
            if k1 not in graph:
                graph[k1] = []
            if k2 not in graph[k1]:
                graph[k1].append(k2)
    
    if not graph:
        return poly_a
    
    # Find starting point (leftmost, then bottommost)
    start = min(graph.keys(), key=lambda k: (k[0], k[1]))
    
    # Trace boundary following outside edges
    result = []
    current = start
    visited = set()
    max_steps = len(point_map) * 4
    step = 0
    
    while step < max_steps:
        if current not in graph or not graph[current]:
            break
        
        # Find next point - prefer leftmost turn (counter-clockwise)
        next_point = None
        best_turn = None
        
        # Calculate incoming direction
        incoming_dir = None
        if len(result) >= 2:
            prev_key = point_to_key(result[-2])
            dx = current[0] - prev_key[0]
            dy = current[1] - prev_key[1]
            incoming_dir = math.atan2(dy, dx)
            if incoming_dir < 0:
                incoming_dir += 2 * math.pi
        
        for candidate in graph[current]:
            edge = (current, candidate)
            if edge in visited:
                continue
            
            # Calculate direction to candidate
            dx = candidate[0] - current[0]
            dy = candidate[1] - current[1]
            angle = math.atan2(dy, dx)
            if angle < 0:
                angle += 2 * math.pi
            
            if incoming_dir is None:
                # First step: prefer going up or right
                if next_point is None:
                    next_point = candidate
                    best_turn = angle
                elif abs(angle - math.pi / 2) < abs(best_turn - math.pi / 2):
                    next_point = candidate
                    best_turn = angle
            else:
                # Calculate turn (positive = left turn, negative = right turn)
                turn = angle - incoming_dir
                if turn > math.pi:
                    turn -= 2 * math.pi
                elif turn < -math.pi:
                    turn += 2 * math.pi
                
                # Prefer smallest positive turn (leftmost)
                if best_turn is None:
                    best_turn = turn
                    next_point = candidate
                elif turn >= 0:
                    if best_turn < 0 or turn < best_turn:
                        best_turn = turn
                        next_point = candidate
                elif best_turn < 0 and turn > best_turn:
                    best_turn = turn
                    next_point = candidate
        
        # Fallback: take any unvisited edge
        if next_point is None:
            for candidate in graph[current]:
                edge = (current, candidate)
                if edge not in visited:
                    next_point = candidate
                    break
        
        if next_point is None:
            break
        
        # Add current point
        if not result or distance(point_map[current], result[-1]) > 1e-6:
            result.append(point_map[current])
        
        # Mark edge as visited
        visited.add((current, next_point))
        current = next_point
        
        # Check completion - need to return to start
        if current == start and len(result) >= 3:
            # Add the start point to close the loop
            if distance(point_map[current], result[-1]) > 1e-6:
                result.append(point_map[current])
            break
        
        step += 1
    
    # If we didn't complete the loop but have enough points, try to close it
    if len(result) >= 3 and current != start:
        # Check if we can reach start from current
        if current in graph and start in graph.get(current, []):
            result.append(point_map[start])
        elif distance(point_map[current], point_map[start]) < 1e-6:
            # Already at start
            pass
        else:
            # Try to find a path back or use convex hull
            if len(result) < 4:
                return convex_hull_union(poly_a, poly_b)
    
    # Clean result
    if not result:
        # Fallback: return convex hull if tracing failed
        return convex_hull_union(poly_a, poly_b)
    
    # Remove duplicates
    cleaned = []
    for p in result:
        if not cleaned or distance(p, cleaned[-1]) > 1e-6:
            cleaned.append(p)
    
    # Ensure we have at least 3 points for a valid polygon
    if len(cleaned) < 3:
        return convex_hull_union(poly_a, poly_b)
    
    # Close polygon if needed
    if len(cleaned) > 2:
        if distance(cleaned[0], cleaned[-1]) > 1e-6:
            cleaned.append(cleaned[0])
        elif len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
            cleaned.pop()
    
    # Final validation - ensure we have a valid polygon
    if len(cleaned) < 3:
        return convex_hull_union(poly_a, poly_b)
    
    return cleaned

def cross_product(o, a, b):
    """
    Calculate cross product of vectors oa and ob.
    
    Used to determine the orientation of three points (for convex hull algorithm).
    Positive result means counter-clockwise turn, negative means clockwise.
    
    Args:
        o (tuple): Origin point (x, y)
        a (tuple): First point (x, y)
        b (tuple): Second point (x, y)
        
    Returns:
        float: Cross product value (positive = counter-clockwise, negative = clockwise)
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def convex_hull_union(poly_a, poly_b):
    """
    Compute convex hull of two disjoint polygons using Graham scan algorithm.
    
    Used as a fallback when polygons don't intersect. The convex hull is the
    smallest convex polygon that contains all points from both polygons.
    
    Algorithm (Graham Scan):
    1. Find the bottom-most (or leftmost) point as the starting point
    2. Sort all other points by polar angle with respect to the start point
    3. Build the hull by iteratively adding points and removing points that
       create clockwise turns (using cross product)
    
    Args:
        poly_a (list): First polygon as list of (x, y) tuples
        poly_b (list): Second polygon as list of (x, y) tuples
        
    Returns:
        list: Convex hull as list of (x, y) tuples
    """
    # Get all unique points from both polygons
    all_points = list(set(poly_a + poly_b))
    
    if len(all_points) < 3:
        return all_points
    
    # Find bottom-most point (or leftmost in case of tie) as starting point
    start = min(all_points, key=lambda p: (p[1], p[0]))
    
    # Sort points by polar angle with respect to start point
    def polar_angle(p):
        """Calculate polar angle of point p relative to start point."""
        dx = p[0] - start[0]
        dy = p[1] - start[1]
        return math.atan2(dy, dx)
    
    sorted_points = sorted([p for p in all_points if p != start], key=polar_angle)
    sorted_points.insert(0, start)
    
    # Build convex hull using Graham scan
    hull = [sorted_points[0], sorted_points[1]]
    
    for i in range(2, len(sorted_points)):
        # Remove points that create clockwise turns (cross product <= 0)
        while len(hull) > 1 and cross_product(hull[-2], hull[-1], sorted_points[i]) <= 0:
            hull.pop()
        hull.append(sorted_points[i])
    
    return hull
  
def intersection(poly_a, poly_b):
    """
    Compute the intersection of two polygons using geometric algorithms.
    
    The intersection operation finds the overlapping region between two polygons,
    including only points that are in both polygons. Uses a boundary tracing
    algorithm that follows the inner edges (edges inside both polygons).
    
    Algorithm:
    1. Find all intersection points between the two polygons
    2. Create enhanced vertex lists with intersections inserted
    3. Build a graph of edges that are inside the other polygon
    4. Trace the boundary following inside edges in clockwise order
    
    Args:
        poly_a (list): First polygon as list of (x, y) tuples
        poly_b (list): Second polygon as list of (x, y) tuples
        
    Returns:
        list: Result polygon as list of (x, y) tuples representing the intersection.
              Returns empty list if polygons don't overlap.
    """
    # Handle empty polygons
    if not poly_a or not poly_b:
        return []
    
    # Find all intersection points
    intersections = get_all_intersections(poly_a, poly_b)
    
    # If no intersections, check if one polygon contains the other
    if not intersections:
        if point_in_polygon(poly_a[0], poly_b):
            return poly_a  # poly_a is inside poly_b
        if point_in_polygon(poly_b[0], poly_a):
            return poly_b  # poly_b is inside poly_a
        # Disjoint polygons - no intersection
        return []
    
    # Create enhanced vertex lists with intersections (same as union)
    def create_enhanced_list(poly, intersections, poly_id):
        n = len(poly)
        enhanced = []
        
        for i in range(n):
            start = poly[i]
            end = poly[(i + 1) % n]
            
            # Find intersections on this edge
            edge_inters = []
            for inter, idx_a, idx_b, id_a, id_b in intersections:
                if id_a == poly_id and idx_a == i:
                    d = distance(start, inter)
                    edge_inters.append((inter, d))
            
            # Sort by distance
            edge_inters.sort(key=lambda x: x[1])
            
            # Add start point (only for first edge, or if different from last)
            if i == 0 or distance(start, enhanced[-1]) > 1e-6:
                enhanced.append(start)
            # Add intersections in order
            for inter, _ in edge_inters:
                if distance(inter, enhanced[-1]) > 1e-6:
                    enhanced.append(inter)
        
        return enhanced
    
    list_a = create_enhanced_list(poly_a, intersections, 'a')
    list_b = create_enhanced_list(poly_b, intersections, 'b')
    
    # Create point to index mapping
    def point_to_key(p):
        return (round(p[0], 8), round(p[1], 8))
    
    # Build connectivity graph for INSIDE edges only (opposite of union)
    graph = {}
    point_map = {}
    
    # Add all points to map
    for p in list_a + list_b:
        key = point_to_key(p)
        point_map[key] = p
    
    # Add edges from poly_a if INSIDE poly_b (for intersection)
    n_a = len(list_a)
    for i in range(n_a):
        p1 = list_a[i]
        p2 = list_a[(i + 1) % n_a]
        k1 = point_to_key(p1)
        k2 = point_to_key(p2)
        
        # Check if edge is inside (for intersection, we want inside edges)
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        if point_in_polygon(mid, poly_b):
            if k1 not in graph:
                graph[k1] = []
            if k2 not in graph[k1]:
                graph[k1].append(k2)
    
    # Add edges from poly_b if INSIDE poly_a (for intersection)
    n_b = len(list_b)
    for i in range(n_b):
        p1 = list_b[i]
        p2 = list_b[(i + 1) % n_b]
        k1 = point_to_key(p1)
        k2 = point_to_key(p2)
        
        # Check if edge is inside (for intersection, we want inside edges)
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        if point_in_polygon(mid, poly_a):
            if k1 not in graph:
                graph[k1] = []
            if k2 not in graph[k1]:
                graph[k1].append(k2)
    
    if not graph:
        return []
    
    # Find starting point (could be any intersection point or inside vertex)
    # Prefer an intersection point if available
    start = None
    for inter, _, _, _, _ in intersections:
        key = point_to_key(inter)
        if key in graph:
            start = key
            break
    
    # If no intersection point in graph, use leftmost point
    if start is None:
        start = min(graph.keys(), key=lambda k: (k[0], k[1]))
    
    # Trace boundary following inside edges
    result = []
    current = start
    visited = set()
    max_steps = len(point_map) * 4
    step = 0
    
    while step < max_steps:
        if current not in graph or not graph[current]:
            break
        
        # Find next point - prefer rightmost turn (clockwise for inner boundary)
        next_point = None
        best_turn = None
        
        # Calculate incoming direction
        incoming_dir = None
        if len(result) >= 2:
            prev_key = point_to_key(result[-2])
            dx = current[0] - prev_key[0]
            dy = current[1] - prev_key[1]
            incoming_dir = math.atan2(dy, dx)
            if incoming_dir < 0:
                incoming_dir += 2 * math.pi
        
        for candidate in graph[current]:
            edge = (current, candidate)
            if edge in visited:
                continue
            
            # Calculate direction to candidate
            dx = candidate[0] - current[0]
            dy = candidate[1] - current[1]
            angle = math.atan2(dy, dx)
            if angle < 0:
                angle += 2 * math.pi
            
            if incoming_dir is None:
                # First step: prefer going down or left (for inner boundary)
                if next_point is None:
                    next_point = candidate
                    best_turn = angle
                elif abs(angle - (3 * math.pi / 2)) < abs(best_turn - (3 * math.pi / 2)):
                    next_point = candidate
                    best_turn = angle
            else:
                # Calculate turn (positive = left turn, negative = right turn)
                turn = angle - incoming_dir
                if turn > math.pi:
                    turn -= 2 * math.pi
                elif turn < -math.pi:
                    turn += 2 * math.pi
                
                # For inner boundary, prefer largest negative turn (rightmost/clockwise)
                if best_turn is None:
                    best_turn = turn
                    next_point = candidate
                elif turn <= 0:
                    if best_turn > 0 or turn > best_turn:
                        best_turn = turn
                        next_point = candidate
                elif best_turn > 0 and turn <= 0:
                    best_turn = turn
                    next_point = candidate
        
        # Fallback: take any unvisited edge
        if next_point is None:
            for candidate in graph[current]:
                edge = (current, candidate)
                if edge not in visited:
                    next_point = candidate
                    break
        
        if next_point is None:
            break
        
        # Add current point
        if not result or distance(point_map[current], result[-1]) > 1e-6:
            result.append(point_map[current])
        
        # Mark edge as visited
        visited.add((current, next_point))
        current = next_point
        
        # Check completion - need to return to start
        if current == start and len(result) >= 3:
            # Add the start point to close the loop
            if distance(point_map[current], result[-1]) > 1e-6:
                result.append(point_map[current])
            break
        
        step += 1
    
    # If we didn't complete the loop but have enough points, try to close it
    if len(result) >= 3 and current != start:
        # Check if we can reach start from current
        if current in graph and start in graph.get(current, []):
            result.append(point_map[start])
        elif distance(point_map[current], point_map[start]) < 1e-6:
            # Already at start
            pass
    
    # Clean result
    if not result:
        return []
    
    # Remove duplicates
    cleaned = []
    for p in result:
        if not cleaned or distance(p, cleaned[-1]) > 1e-6:
            cleaned.append(p)
    
    # Ensure we have at least 3 points for a valid polygon
    if len(cleaned) < 3:
        return []
    
    # Close polygon if needed
    if len(cleaned) > 2:
        if distance(cleaned[0], cleaned[-1]) > 1e-6:
            cleaned.append(cleaned[0])
        elif len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
            cleaned.pop()
    
    # Final validation
    if len(cleaned) < 3:
        return []
    
    return cleaned

def difference(poly_a, poly_b):
    """
    Compute the difference of two polygons (A - B): parts of A that are not in B.
    
    The difference operation subtracts polygon B from polygon A, resulting in
    all points that are in A but not in B. This can create polygons with holes
    if B is partially inside A.
    
    Algorithm:
    1. Find all intersection points between the two polygons
    2. Create enhanced vertex lists with intersections inserted
    3. Build a graph of:
       - Edges from A that are outside B (part of result)
       - Edges from B that are inside A (traced in reverse to form hole boundaries)
    4. Trace the boundary following outside edges of A
    
    Args:
        poly_a (list): First polygon (A) as list of (x, y) tuples
        poly_b (list): Second polygon (B) as list of (x, y) tuples
        
    Returns:
        list: Result polygon as list of (x, y) tuples representing A - B.
              Returns empty list if A is completely inside B.
    """
    # Handle empty polygons
    if not poly_a:
        return []
    if not poly_b:
        return poly_a  # A - empty = A
    
    # Find all intersection points
    intersections = get_all_intersections(poly_a, poly_b)
    
    # If no intersections, check if A is inside B
    if not intersections:
        if point_in_polygon(poly_a[0], poly_b):
            return []  # A is completely inside B, so A - B = empty
        # A and B are disjoint, so A - B = A
        return poly_a
    
    # Create enhanced vertex lists with intersections
    def create_enhanced_list(poly, intersections, poly_id):
        n = len(poly)
        enhanced = []
        
        for i in range(n):
            start = poly[i]
            end = poly[(i + 1) % n]
            
            # Find intersections on this edge
            edge_inters = []
            for inter, idx_a, idx_b, id_a, id_b in intersections:
                if id_a == poly_id and idx_a == i:
                    d = distance(start, inter)
                    edge_inters.append((inter, d))
            
            # Sort by distance
            edge_inters.sort(key=lambda x: x[1])
            
            # Add start point (only for first edge, or if different from last)
            if i == 0 or distance(start, enhanced[-1]) > 1e-6:
                enhanced.append(start)
            # Add intersections in order
            for inter, _ in edge_inters:
                if distance(inter, enhanced[-1]) > 1e-6:
                    enhanced.append(inter)
        
        return enhanced
    
    list_a = create_enhanced_list(poly_a, intersections, 'a')
    list_b = create_enhanced_list(poly_b, intersections, 'b')
    
    # Create point to index mapping
    def point_to_key(p):
        return (round(p[0], 8), round(p[1], 8))
    
    # Build connectivity graph for difference (A - B)
    # Include edges from A that are outside B
    # Include edges from B that are outside A (to trace around holes)
    graph = {}
    point_map = {}
    
    # Add all points to map
    for p in list_a + list_b:
        key = point_to_key(p)
        point_map[key] = p
    
    # Add edges from poly_a if outside poly_b (these are part of A - B)
    n_a = len(list_a)
    for i in range(n_a):
        p1 = list_a[i]
        p2 = list_a[(i + 1) % n_a]
        k1 = point_to_key(p1)
        k2 = point_to_key(p2)
        
        # Check if edge is outside B (for difference, we want A edges outside B)
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        if not point_in_polygon(mid, poly_b):
            if k1 not in graph:
                graph[k1] = []
            if k2 not in graph[k1]:
                graph[k1].append(k2)
    
    # Add edges from poly_b if outside poly_a (to trace around the "hole" created by B in A)
    # But we need to reverse direction to trace the inner boundary
    n_b = len(list_b)
    for i in range(n_b):
        p1 = list_b[i]
        p2 = list_b[(i + 1) % n_b]
        k1 = point_to_key(p1)
        k2 = point_to_key(p2)
        
        # Check if edge is inside A (this forms the boundary of the "hole")
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        if point_in_polygon(mid, poly_a):
            # Add in reverse direction to trace the inner boundary
            if k2 not in graph:
                graph[k2] = []
            if k1 not in graph[k2]:
                graph[k2].append(k1)
    
    if not graph:
        # If no graph, check if A is completely inside B
        if point_in_polygon(poly_a[0], poly_b):
            return []
        return poly_a
    
    # Find starting point - prefer a point from A that's outside B
    start = None
    for p in poly_a:
        key = point_to_key(p)
        if key in graph and not point_in_polygon(p, poly_b):
            start = key
            break
    
    # If no suitable point from A, use leftmost point in graph
    if start is None:
        start = min(graph.keys(), key=lambda k: (k[0], k[1]))
    
    # Trace boundary following outside edges of A
    result = []
    current = start
    visited = set()
    max_steps = len(point_map) * 4
    step = 0
    
    while step < max_steps:
        if current not in graph or not graph[current]:
            break
        
        # Find next point - prefer leftmost turn (counter-clockwise for outer boundary)
        next_point = None
        best_turn = None
        
        # Calculate incoming direction
        incoming_dir = None
        if len(result) >= 2:
            prev_key = point_to_key(result[-2])
            dx = current[0] - prev_key[0]
            dy = current[1] - prev_key[1]
            incoming_dir = math.atan2(dy, dx)
            if incoming_dir < 0:
                incoming_dir += 2 * math.pi
        
        for candidate in graph[current]:
            edge = (current, candidate)
            if edge in visited:
                continue
            
            # Calculate direction to candidate
            dx = candidate[0] - current[0]
            dy = candidate[1] - current[1]
            angle = math.atan2(dy, dx)
            if angle < 0:
                angle += 2 * math.pi
            
            if incoming_dir is None:
                # First step: prefer going up or right
                if next_point is None:
                    next_point = candidate
                    best_turn = angle
                elif abs(angle - math.pi / 2) < abs(best_turn - math.pi / 2):
                    next_point = candidate
                    best_turn = angle
            else:
                # Calculate turn (positive = left turn, negative = right turn)
                turn = angle - incoming_dir
                if turn > math.pi:
                    turn -= 2 * math.pi
                elif turn < -math.pi:
                    turn += 2 * math.pi
                
                # Prefer smallest positive turn (leftmost) for outer boundary
                if best_turn is None:
                    best_turn = turn
                    next_point = candidate
                elif turn >= 0:
                    if best_turn < 0 or turn < best_turn:
                        best_turn = turn
                        next_point = candidate
                elif best_turn < 0 and turn >= 0:
                    best_turn = turn
                    next_point = candidate
        
        # Fallback: take any unvisited edge
        if next_point is None:
            for candidate in graph[current]:
                edge = (current, candidate)
                if edge not in visited:
                    next_point = candidate
                    break
        
        if next_point is None:
            break
        
        # Add current point
        if not result or distance(point_map[current], result[-1]) > 1e-6:
            result.append(point_map[current])
        
        # Mark edge as visited
        visited.add((current, next_point))
        current = next_point
        
        # Check completion - need to return to start
        if current == start and len(result) >= 3:
            # Add the start point to close the loop
            if distance(point_map[current], result[-1]) > 1e-6:
                result.append(point_map[current])
            break
        
        step += 1
    
    # If we didn't complete the loop but have enough points, try to close it
    if len(result) >= 3 and current != start:
        # Check if we can reach start from current
        if current in graph and start in graph.get(current, []):
            result.append(point_map[start])
        elif distance(point_map[current], point_map[start]) < 1e-6:
            # Already at start
            pass
    
    # Clean result
    if not result:
        return []
    
    # Remove duplicates
    cleaned = []
    for p in result:
        if not cleaned or distance(p, cleaned[-1]) > 1e-6:
            cleaned.append(p)
    
    # Ensure we have at least 3 points for a valid polygon
    if len(cleaned) < 3:
        return []
    
    # Close polygon if needed
    if len(cleaned) > 2:
        if distance(cleaned[0], cleaned[-1]) > 1e-6:
            cleaned.append(cleaned[0])
        elif len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
            cleaned.pop()
    
    # Final validation
    if len(cleaned) < 3:
        return []
    
    return cleaned

# ---- Polygon Management ----

def main():
    """
    Main program loop for polygon management and set operations.
    
    Provides an interactive menu system for:
    - Adding polygons (from keyboard or file)
    - Saving polygons to files
    - Displaying polygons
    - Performing set operations (union, intersection, difference)
    - Removing polygons
    
    The program maintains a dictionary of polygons keyed by name,
    allowing users to perform operations on named polygon objects.
    """
    polygons = {}  # Dictionary to store polygons: {name: [(x, y), ...]}
    while True:
        print("\n1. Add polygon (keyboard/file)")
        print("2. Save polygon to file")
        print("3. Show polygons")
        print("4. Combine (union/intersection/difference)")
        print("5. Remove polygon")
        print("6. Quit")
        choice = input("Choice: ")

        if choice == '1':
            # Add polygon: Get name and input method (keyboard or file)
            name = input("Enter polygon name: ")
            method = input("Enter 'k' for keyboard, 'f' for file: ")
            if method == 'k':
                polygons[name] = read_polygon_from_input()
            elif method == 'f':
                fname = input("Filename: ")
                polygons[name] = read_polygon_from_file(fname)
        elif choice == '2':
            # Save polygon to file
            name = input("Enter polygon name to save: ")
            fname = input("Enter filename: ")
            if name in polygons:
                write_polygon_to_file(polygons[name], fname)
            else:
                print(f"Error: Polygon '{name}' not found.")
        elif choice == '3':
            # Display all polygons
            if polygons:
                display_polygons(list(polygons.values()), list(polygons.keys()))
            else:
                print("No polygons to display.")
        elif choice == '4':
            print(list(polygons.keys()))
            op = input("Enter operation (u/U - Union/ i/I - Intersection/ d/D - Difference): ").lower()
            
            if op == 'union' or op == 'u':
                a = input("Name first polygon: ")
                b = input("Name second polygon: ")
                if a not in polygons or b not in polygons:
                    print("Error: One or both polygons not found.")
                else:
                    resultname = input("Result polygon name: ")
                    result = union(polygons[a], polygons[b])
                    if result and len(result) >= 3:
                        polygons[resultname] = result
                        print(f"Union operation completed. Result polygon '{resultname}' has {len(result)} vertices.")
                        # Automatically display all polygons including the result
                        # Highlight the result polygon with thicker line
                        poly_list = list(polygons.values())
                        label_list = list(polygons.keys())
                        result_index = label_list.index(resultname)
                        display_polygons(poly_list, label_list, highlight_index=result_index)
                    else:
                        print(f"Error: Union operation failed or returned invalid result.")
            elif op == 'intersection' or op == 'i':
                a = input("Name first polygon: ")
                b = input("Name second polygon: ")
                if a not in polygons or b not in polygons:
                    print("Error: One or both polygons not found.")
                else:
                    resultname = input("Result polygon name: ")
                    result = intersection(polygons[a], polygons[b])
                    if result and len(result) >= 3:
                        polygons[resultname] = result
                        print(f"Intersection operation completed. Result polygon '{resultname}' has {len(result)} vertices.")
                        # Automatically display all polygons including the result
                        # Highlight the result polygon with thicker line
                        poly_list = list(polygons.values())
                        label_list = list(polygons.keys())
                        result_index = label_list.index(resultname)
                        display_polygons(poly_list, label_list, highlight_index=result_index)
                    else:
                        print(f"Error: Intersection operation failed or returned empty result (polygons may not overlap).")
                        # Still display the input polygons
                        display_polygons(list(polygons.values()), list(polygons.keys()))
            elif op == 'difference' or op == 'd':
                a = input("Difference from ? ")
                b = input("Difference with ? ")
                if a not in polygons or b not in polygons:
                    print("Error: One or both polygons not found.")
                else:
                    resultname = input("Result polygon name: ")
                    result = difference(polygons[a], polygons[b])
                    if result and len(result) >= 3:
                        polygons[resultname] = result
                        print(f"Difference operation completed. Result polygon '{resultname}' has {len(result)} vertices.")
                        # Automatically display all polygons including the result
                        # Highlight the result polygon with thicker line
                        poly_list = list(polygons.values())
                        label_list = list(polygons.keys())
                        result_index = label_list.index(resultname)
                        display_polygons(poly_list, label_list, highlight_index=result_index)
                    else:
                        print(f"Error: Difference operation failed or returned empty result.")
                        # Still display the input polygons
                        display_polygons(list(polygons.values()), list(polygons.keys()))
            else:
                print("Invalid operation.")
        elif choice == '5':
            # Remove polygon by name
            if not polygons:
                print("No polygons to remove.")
            else:
                print(f"Available polygons: {list(polygons.keys())}")
                name = input("Enter polygon name to remove: ")
                if name in polygons:
                    del polygons[name]
                    print(f"Polygon '{name}' has been removed.")
                else:
                    print(f"Error: Polygon '{name}' not found.")
        elif choice == '6':
            # Quit program
            break
        else:
            print("Invalid choice.")

        input("Press Enter to continue...")

if __name__ == '__main__':
    main()
