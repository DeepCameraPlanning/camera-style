"""
This script comments line 22 and 23 of the file lib/sort/sort.py, in order to
avoid dependency issues.
"""

if __name__ == "__main__":
    file_path = "lib/sort/sort.py"

    # Comment line 22
    with open(file_path, "r+") as f:
        file_content = f.read()
        f.seek(832)
        f.write("# " + file_content[832:])

    # Comment line 23
    with open(file_path, "r+") as f:
        file_content = f.read()
        f.seek(852)
        f.write("# " + file_content[852:])
