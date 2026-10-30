import sys


def main():
    path = sys.argv[1]
    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f):
            print(line.rstrip())
            if i >= 9:
                break


if __name__ == "__main__":
    main()
