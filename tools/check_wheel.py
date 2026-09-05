"""Check that built wheels include each URDF's referenced mesh files."""

import argparse
import posixpath
from pathlib import PurePosixPath
from xml.etree import ElementTree
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+")
    args = parser.parse_args()
    for path in args.wheels:
        with ZipFile(path) as wheel:
            files = set(wheel.namelist())
            models = sorted(name for name in files if name.endswith(".urdf"))
            if not models:
                raise RuntimeError(f"{path}: no bundled URDF")
            count = 0
            for name in models:
                for mesh in ElementTree.fromstring(wheel.read(name)).findall(".//mesh"):
                    resource = posixpath.normpath(
                        str(PurePosixPath(name).parent / mesh.attrib["filename"])
                    )
                    if resource not in files:
                        raise RuntimeError(f"{path}: missing mesh {resource}")
                    count += 1
            print(f"{path}: {len(models)} URDF, {count} mesh references verified")


if __name__ == "__main__":
    main()
