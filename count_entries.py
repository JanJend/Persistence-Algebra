import sys

def count_numbers_after_semicolon(filename):
    count = 0
    with open(filename, "r") as f:
        for line in f:
            if ';' in line:
                parts = line.split(';')[1].strip()
                if parts:
                    numbers = parts.split()
                    count += len(numbers)
    return count

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_semicolon_numbers.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    total = count_numbers_after_semicolon(filename)
    print(f"Total numbers after semicolons: {total}")