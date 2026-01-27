def square(side):
    print("Area of the square: {}".format(side * side))


def rectangle(width, height):
    print("Area of the rectangle: {}".format(width * height))


def right_triangle(base, height):
    area = (base * height) / 2
    print("Area of the right triangle: {}".format(area))


def parallelogram(base, height):
    print("Area of the parallelogram: {}".format(base * height))


while True:
    print("""
    0: Exit
    1: Square
    2: Rectangle
    3: Right Triangle
    4: Parallelogram
    """)

    choice = int(input("Select the geometric shape to calculate its area: "))

    if choice == 1:
        side = int(input("Enter the side length of the square: "))
        square(side)

    elif choice == 2:
        short_side = int(input("Enter the short side of the rectangle: "))
        long_side = int(input("Enter the long side of the rectangle: "))
        rectangle(long_side, short_side)

    elif choice == 3:
        base = int(input("Enter the base length of the triangle: "))
        height = int(input("Enter the height of the triangle: "))
        right_triangle(base, height)

    elif choice == 4:
        base = int(input("Enter the base length of the parallelogram: "))
        height = int(input("Enter the height of the parallelogram: "))
        parallelogram(base, height)

    elif choice == 0:
        break
