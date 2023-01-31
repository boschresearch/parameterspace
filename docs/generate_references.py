"""Generate "virtual" doc files for the api references.

The files are only generated during build time and not actually written on disk.
To write them on disk (for e.g. debugging) execute this script directely.
"""
from pathlib import Path

import mkdocs_gen_files

# Files to exclude from the generated API docs
excludes = ["__init__"]

src_root = Path("parameterspace")
for path in src_root.glob("**/*.py"):
    doc_path = Path("API-Reference", path.relative_to(src_root)).with_suffix(".md")

    if any(exclude in str(doc_path) for exclude in excludes):
        continue

    with mkdocs_gen_files.open(doc_path, "w") as f:
        ident = ".".join(path.with_suffix("").parts)
        print("::: " + ident, file=f)
