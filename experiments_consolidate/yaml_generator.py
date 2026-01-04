# generate_environment_yaml.py
import ast
import importlib.metadata as md
import nbformat
import os
import sys

# --- CONFIG ---------------------------------------------------------

ENV_NAME = "simon_data_gen"
OUTPUT_YAML = "environment.yaml"

# directories to skip while walking
IGNORE_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".ipynb_checkpoints",
    "venv",
    ".venv",
    "env",
    ".mypy_cache",
}


# --- PYTHON VERSION (AUTO) ------------------------------------------

py_major = sys.version_info.major
py_minor = sys.version_info.minor
PYTHON_VERSION = f"{py_major}.{py_minor}"


# --- HELPERS --------------------------------------------------------


def extract_imports_from_code(code: str) -> set[str]:
    """Parse Python code and return imported top-level module names."""
    tree = ast.parse(code)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    return imports


def imports_from_py(path: str) -> set[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        return extract_imports_from_code(code)
    except Exception as e:
        print(f"⚠️  Skipping {path} (py) due to error: {e}")
        return set()


def imports_from_ipynb(path: str) -> set[str]:
    try:
        nb = nbformat.read(path, as_version=4)
        out: set[str] = set()
        for cell in nb.cells:
            if cell.cell_type == "code":
                try:
                    out |= extract_imports_from_code(cell.source)
                except SyntaxError:
                    # ignore cells with magics / invalid syntax
                    pass
        return out
    except Exception as e:
        print(f"⚠️  Skipping {path} (ipynb) due to error: {e}")
        return set()


# --- COLLECT IMPORTS (RECURSIVE SCAN) -------------------------------

all_imports: set[str] = set()

for root, dirs, files in os.walk("."):
    # prune ignored dirs in-place
    dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

    for filename in files:
        full_path = os.path.join(root, filename)

        if filename.endswith(".py"):
            all_imports |= imports_from_py(full_path)
        elif filename.endswith(".ipynb"):
            all_imports |= imports_from_ipynb(full_path)


# --- DETECT LOCAL MODULES (YOUR OWN FILES) --------------------------

local_modules: set[str] = set()
for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

    for fname in files:
        if fname.endswith(".py"):
            mod_name = os.path.splitext(fname)[0]
            local_modules.add(mod_name)

# --- STD-LIB-LIKE NAMES TO SKIP ------------------------------------

stdlib_like = {
    "__future__",
    "ast",
    "importlib",
    "os",
    "pathlib",
    "sys",
    "time",
    "typing",
    "builtins",
    "types",
}


# --- RESOLVE VERSIONS FOR REAL PACKAGES -----------------------------

reqs: list[str] = []

for pkg in sorted(all_imports):
    # skip stdlib, dunder stuff, and local modules like data_handling
    if pkg in stdlib_like or pkg in local_modules or pkg.startswith("_"):
        continue

    try:
        version = md.version(pkg)
        reqs.append(f"{pkg}=={version}")
    except md.PackageNotFoundError:
        # likely a non-distribution import (namespace package, etc.)
        print(f"ℹ️  Skipping non-distribution import: {pkg}")


# --- WRITE environment.yaml ----------------------------------------

yaml_lines = [
    f"name: {ENV_NAME}",
    "dependencies:",
    f"  - python={PYTHON_VERSION}",
    "  - pip",
    "  - pip:",
]

for r in reqs:
    yaml_lines.append(f"      - {r}")

yaml_content = "\n".join(yaml_lines) + "\n"

with open(OUTPUT_YAML, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(
    f"✅ Wrote {OUTPUT_YAML} "
    f"with python={PYTHON_VERSION} and {len(reqs)} pip dependencies"
)
