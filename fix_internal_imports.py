#!/usr/bin/env python3
import ast
import os
from pathlib import Path

# --- Configurazione dinamica della root ---
SCRIPT_PATH    = Path(__file__).resolve()
PROJECT_ROOT   = SCRIPT_PATH.parent            # cartella che contiene 'src'
SRC_IMMOBILIARE = PROJECT_ROOT / "src" / "immobiliare"
MODULE_ROOT    = "immobiliare"                 # prefisso Python per import interni

def discover_internal_modules(root: Path):
    """
    Scansiona src/immobiliare per elencare
    i moduli/pacchetti interni (es. 'cli', 'pipeline', 'utils', ...)
    """
    modules = set()
    for item in root.iterdir():
        if item.is_dir() and (item / "__init__.py").exists():
            modules.add(item.name)
        elif item.is_file() and item.suffix == ".py":
            modules.add(item.stem)
    return modules

INTERNAL = discover_internal_modules(SRC_IMMOBILIARE)

def fix_imports_in_code(source: str):
    """
    Usa ast per trovare gli import assoluti di moduli interni
    e li riscrive con prefisso MODULE_ROOT.
    """
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    changed = False

    for node in ast.walk(tree):
        # import X
        if isinstance(node, ast.Import):
            for alias in node.names:
                name0 = alias.name.split(".")[0]
                if name0 in INTERNAL:
                    old = f"import {alias.name}"
                    new = f"import {MODULE_ROOT}.{alias.name}"
                    lines = [l.replace(old, new) if l.strip().startswith(old) else l for l in lines]
                    changed = True

        # from X import Y
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                name0 = node.module.split(".")[0]
                if name0 in INTERNAL:
                    old = f"from {node.module} import"
                    new = f"from {MODULE_ROOT}.{node.module} import"
                    lines = [l.replace(old, new) if l.strip().startswith(f"from {node.module}") else l for l in lines]
                    changed = True

    return changed, "".join(lines)

def process_file(py_path: Path):
    src = py_path.read_text(encoding="utf-8")
    changed, new_src = fix_imports_in_code(src)
    if changed:
        py_path.write_text(new_src, encoding="utf-8")
    return changed

def main():
    modified = []
    for py in SRC_IMMOBILIARE.rglob("*.py"):
        if process_file(py):
            # calcola path relativo alla root del progetto per reporting
            rel = py.relative_to(PROJECT_ROOT)
            modified.append(str(rel))
    if modified:
        print("✏️  Corrette importazioni nei seguenti file:")
        for f in modified:
            print("  -", f)
    else:
        print("✅ Nessuna importazione interna da correggere.")

if __name__ == "__main__":
    main()
