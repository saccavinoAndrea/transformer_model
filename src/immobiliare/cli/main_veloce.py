import os
import shutil
import zipfile
from pathlib import Path
import tempfile
import ast
from collections import defaultdict

# 🔧 CONFIGURAZIONE
ORIGINAL_PROJECT_PATH = Path("C:/Users/Andrea/Desktop/scraping_LLM/PythonProject")  # <-- Cambia qui se vuoi

EXCLUDED_DIRS = {".venv", "data"}

def is_excluded(path: Path, root: Path) -> bool:
    """
    Controlla se il path o una sua cartella padre è da escludere.
    """
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        # path non relativo a root, non escludere per sicurezza
        return False
    return any(part in EXCLUDED_DIRS for part in relative_parts)

def copy_project(src: Path, dst: Path):
    for root, dirs, files in os.walk(src):
        root_path = Path(root)
        if is_excluded(root_path, src):
            # Escludo tutta la dir e non scendo dentro
            dirs[:] = []
            continue
        rel_root = root_path.relative_to(src)
        target_dir = dst / rel_root
        target_dir.mkdir(parents=True, exist_ok=True)
        # Filtra dirs per escludere quelle indesiderate anche a livello di sottocartelle
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for f in files:
            src_file = root_path / f
            if not is_excluded(src_file, src):
                dst_file = target_dir / f
                shutil.copy2(src_file, dst_file)

def add_headers_and_init(temp_project_path: Path):
    for dirpath, _, filenames in os.walk(temp_project_path):
        dir_path = Path(dirpath)
        py_modules = []
        for fname in filenames:
            file_path = dir_path / fname
            rel_path = file_path.relative_to(temp_project_path)
            if fname.endswith(".py"):
                header = f"# {rel_path.as_posix()}\n"
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                if not lines or lines[0] != header:
                    lines.insert(0, header)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                if fname != "__init__.py":
                    modname = fname[:-3]
                    py_modules.append(modname)
        if py_modules:
            init_path = dir_path / "__init__.py"
            rel_pkg = dir_path.relative_to(temp_project_path).as_posix()
            header = f"# {rel_pkg}/__init__.py\n"
            body = "\n".join(f"from .{m} import {m}" for m in sorted(py_modules)) + "\n"
            with open(init_path, 'w', encoding='utf-8') as f:
                f.write(header + body)

def zip_project(project_path: Path, zip_name: str):
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(project_path):
            for f in files:
                full_path = Path(root) / f
                rel_path = full_path.relative_to(project_path)
                zf.write(full_path, arcname=rel_path)
    print(f"✅ Archivio creato: {zip_name}")

# --- Parte per generazione init.py basata su AST (non modificata) ---

PROJECT_ROOT = ORIGINAL_PROJECT_PATH  # Usa lo stesso percorso della copia

def is_private(name: str) -> bool:
    return name.startswith("_")

def extract_public_symbols(file_path: Path):
    """Estrai classi e funzioni pubbliche da un file Python."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return [], []  # File non valido

    classes = []
    functions = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not is_private(node.name):
            classes.append(node.name)
        elif isinstance(node, ast.FunctionDef) and not is_private(node.name):
            functions.append(node.name)
    return classes, functions

def collect_package_structure(project_root: Path):
    """Crea una mappa dei package e dei relativi file .py con le classi/funzioni pubbliche da importare."""
    package_map = defaultdict(list)

    for dirpath, dirs, filenames in os.walk(project_root):
        dir_path = Path(dirpath)
        # Escludo anche qui eventuali cartelle di cache o da escludere
        if "__pycache__" in dir_path.parts or any(part in EXCLUDED_DIRS for part in dir_path.parts):
            continue
        if not any(f.endswith(".py") for f in filenames):
            continue

        # filtro dirs per evitare di scendere in cartelle escluse
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

        for fname in filenames:
            if fname == "__init__.py" or not fname.endswith(".py"):
                continue
            module_path = dir_path / fname
            classes, functions = extract_public_symbols(module_path)
            if classes or functions:
                module_name = fname[:-3]
                package_map[dir_path].append((module_name, classes, functions))

    return package_map

def generate_init_file(init_path: Path, module_data):
    """Genera il contenuto ordinato di un file __init__.py."""
    lines = [f"# {init_path.relative_to(PROJECT_ROOT).as_posix()}\n"]

    imports = []
    for module_name, classes, functions in sorted(module_data, key=lambda x: x[0]):
        if classes:
            for cls in sorted(classes):
                imports.append(f"from .{module_name} import {cls}")
        elif functions:
            for fn in sorted(functions):
                imports.append(f"from .{module_name} import {fn}")

    if not imports:
        return  # Niente da importare

    imports = sorted(set(imports))
    lines += imports
    lines.append("")

    with open(init_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    default_zip_name = "project_updated.zip"
    user_input = input(f"📦 Inserisci il nome del file ZIP da creare [{default_zip_name}]: ").strip()
    zip_name = user_input if user_input else default_zip_name

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_project = Path(tmpdir) / "copied_project"
        print("📁 Copio progetto in cartella temporanea...")
        copy_project(ORIGINAL_PROJECT_PATH, temp_project)

        print("🧠 Applico intestazioni e rigenero __init__.py...")
        add_headers_and_init(temp_project)

        print("🔍 Analisi dei package...")
        global PROJECT_ROOT
        PROJECT_ROOT = temp_project  # Aggiorno radice per generazione init.py
        package_map = collect_package_structure(temp_project)

        print("⚙️ Generazione dei file __init__.py...")
        for package_path, modules in package_map.items():
            init_file = package_path / "__init__.py"
            generate_init_file(init_file, modules)
            print(f"✅ Creato: {init_file.relative_to(temp_project)}")

        print("📦 Comprimo il progetto...")
        zip_project(temp_project, zip_name)

if __name__ == "__main__":
    main()
