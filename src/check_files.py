import sys
from data import parse_line


def check_file(path):
    total = 0
    attacks = 0
    first_attack_line = None
    with open(path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip() == "":
                continue
            try:
                row = parse_line(line)
            except ValueError as e:
                print(f"  BAD LINE {line_number}: {e}")
                print(f"  {line.strip()}")
                return
            total += 1
            if row[-1] == 1:
                attacks += 1
                if first_attack_line is None:
                    first_attack_line = line_number
    print(f"  rows: {total}")
    print(f"  normal (R): {total - attacks}")
    print(f"  attack (T): {attacks}")
    print(f"  first attack at line: {first_attack_line}")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(path)
        check_file(path)
