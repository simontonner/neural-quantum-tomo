# %%
import ast, importlib.metadata as md, os

# --- CONFIG ---
target_file = "data_generation.py"           # file to scan
output_yaml = "environment.yaml"
env_name = "viet_test"

# --- PARSE IMPORTS ---
with open(target_file, "r", encoding="utf-8") as f:
    tree = ast.parse(f.read(), filename=target_file)

imports = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        for alias in node.names:
            imports.add(alias.name.split(".")[0])
    elif isinstance(node, ast.ImportFrom) and node.module:
        imports.add(node.module.split(".")[0])

# --- GET INSTALLED VERSIONS ---
reqs = []
for pkg in sorted(imports):
    try:
        version = md.version(pkg)
        reqs.append(f"{pkg}=={version}")
    except md.PackageNotFoundError:
        print(f"⚠️ Not installed in this environment: {pkg}")

# --- WRITE YAML ---
yaml_content = f"""name: {env_name}
dependencies:
  - python=3.11
  - pip
  - pip:
"""

for r in reqs:
    yaml_content += f"      - {r}\n"

with open(output_yaml, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"✅ Wrote environment.yaml with {len(reqs)} pip dependencies")
print("\n--- environment.yaml ---\n")
print(yaml_content)
