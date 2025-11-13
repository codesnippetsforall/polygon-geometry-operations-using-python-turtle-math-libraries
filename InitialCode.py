import turtle
import math

# ---- Polygon Utilities ----

def read_polygon_from_input():
    '''Get a list of (x, y) tuples from keyboard input.'''
    n = int(input("Enter number of vertices: "))
    points = []
    for i in range(n):
        x, y = map(float, input(f"Enter vertex {i+1} (x y): ").split())
        points.append((x, y))
    return points

def read_polygon_from_file(filename):
    '''Read polygon vertices from a text file. Each line: x y'''
    with open(filename) as f:
        points = [tuple(map(float, line.split())) for line in f]
    return points

def write_polygon_to_file(points, filename):
    '''Save polygon vertices to a file.'''
    with open(filename, 'w') as f:
        for x, y in points:
            f.write(f"{x} {y}\n")

def display_polygons(polygons, labels):
    '''Display multiple polygons in Turtle.'''
    turtle.clearscreen()
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    for i, pts in enumerate(polygons):
        turtle.penup()
        turtle.pencolor(colors[i % len(colors)])
        turtle.goto(pts[0])
        turtle.pendown()
        for pt in pts[1:]:
            turtle.goto(pt)
        turtle.goto(pts[0])  # close the shape
        turtle.penup()
        turtle.goto(pts[0])
        turtle.write(labels[i], font=('Arial', 12, 'bold'))
    turtle.hideturtle()

# ---- Set Operations (Basis for Pseudocode) ----

def union(poly_a, poly_b):
    '''Placeholder for polygon union algorithm from Appendix A.'''
    # -- Replace this with actual geometric algorithm --
    return poly_a  # Placeholder
  
def intersection(poly_a, poly_b):
    '''Placeholder for intersection operation.'''
    return poly_a  # Placeholder

def difference(poly_a, poly_b):
    '''Placeholder for A minus B.'''
    return poly_a  # Placeholder

# ---- Polygon Management ----

def main():
    polygons = {}
    while True:
        print("\n1. Add polygon (keyboard/file)")
        print("2. Save polygon to file")
        print("3. Show polygons")
        print("4. Combine (union/intersection/difference)")
        print("5. Quit")
        choice = input("Choice: ")

        if choice == '1':
            name = input("Enter polygon name: ")
            method = input("Enter 'k' for keyboard, 'f' for file: ")
            if method == 'k':
                polygons[name] = read_polygon_from_input()
            elif method == 'f':
                fname = input("Filename: ")
                polygons[name] = read_polygon_from_file(fname)
        elif choice == '2':
            name = input("Enter polygon name to save: ")
            fname = input("Enter filename: ")
            if name in polygons:
                write_polygon_to_file(polygons[name], fname)
        elif choice == '3':
            display_polygons(list(polygons.values()), list(polygons.keys()))
        elif choice == '4':
            print(list(polygons.keys()))
            a = input("Name first polygon: ")
            b = input("Name second polygon: ")
            op = input("Enter operation (union/intersection/difference): ")
            resultname = input("Result polygon name: ")
            if op == 'union':
                polygons[resultname] = union(polygons[a], polygons[b])
            elif op == 'intersection':
                polygons[resultname] = intersection(polygons[a], polygons[b])
            elif op == 'difference':
                polygons[resultname] = difference(polygons[a], polygons[b])
            else:
                print("Invalid operation.")
        elif choice == '5':
            break
        else:
            print("Invalid choice.")

        input("Press Enter to continue...")

if __name__ == '__main__':
    main()
