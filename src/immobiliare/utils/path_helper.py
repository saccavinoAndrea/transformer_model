import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def timestamped_path(
    path: str | Path,
    *,
    tz: str = "Europe/Rome",
    create_dirs: bool = True,
) -> Path:
    """
    Restituisce un Path con timestamp (_YYYYMMDD_HHMMSS) inserito tra
    nome file ed estensione.  Esempio:

        preprocess/artifacts/normalizer.pkl
    →   preprocess/artifacts/normalizer_20250714_214205.pkl

    Parametri
    ----------
    path : str | Path
        Path "base" del file da versionare.
    tz : str, default "Europe/Rome"
        Timezone IANA per generare il timestamp.
    create_dirs : bool, default True
        Se True crea la directory padre (`mkdir(parents=True, exist_ok=True)`).

    Ritorna
    -------
    Path
        Percorso finale con timestamp e, se richiesto, directory già creata.
    """
    base_path = Path(path)
    timestamp = datetime.now(ZoneInfo(tz)).strftime("%Y%m%d_%H%M%S")
    final_path = base_path.with_name(f"{base_path.stem}_{timestamp}{base_path.suffix}")

    if create_dirs:
        final_path.parent.mkdir(parents=True, exist_ok=True)

    return final_path



# ------------------------------------------------------------
# Helper: risolve il file versionato più recente se necessario
# ------------------------------------------------------------
_TS_RE = re.compile(r"_(\d{8}_\d{6})$")            # _YYYYMMDD_HHMMSS

def resolve_versioned_jsonl(path_like: str | Path) -> Path:
    """
    Se `path_like` punta a un file esistente → lo restituisce.
    Altrimenti:
      1. Usa directory e stem di `path_like` per cercare candidati
         del tipo  <stem>_<timestamp>.jsonl
      2. Se presenti, restituisce il più recente (timestamp max).
      3. Se nessuno trovato, solleva FileNotFoundError.
    """
    p = Path(path_like)
    if p.exists():
        return p

    directory = p.parent if p.parent != Path("") else Path.cwd()
    stem = p.stem                        # senza estensione
    suffix = p.suffix or ".jsonl"

    candidates = list(directory.glob(f"{stem}_*{suffix}"))
    if not candidates:
        raise FileNotFoundError(f"Nessun file versionato per '{stem}' in {directory}")

    def _ts(file_path: Path) -> datetime:
        m = _TS_RE.search(file_path.stem)
        return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S") if m else datetime.min

    return max(candidates, key=_ts)



