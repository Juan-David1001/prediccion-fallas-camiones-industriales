"""Descarga los datos oficiales UCI APS. Ejecutar desde cualquier directorio."""
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile, BadZipFile
import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw"
URL = "https://archive.ics.uci.edu/static/public/421/aps+failure+at+scania+trucks.zip"
NAMES = {"aps_failure_training_set.csv", "aps_failure_test_set.csv"}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    if all((DEST / name).exists() for name in NAMES):
        print("Archivos oficiales existentes en", DEST)
        return
    print("Descargando dataset UCI...")
    response = requests.get(URL, timeout=120)
    response.raise_for_status()
    extracted = set()
    with ZipFile(BytesIO(response.content)) as archive:
        for member in archive.infolist():
            name = Path(member.filename).name
            if name not in NAMES or member.is_dir():
                continue
            (DEST / name).write_bytes(archive.read(member))
            extracted.add(name)
    missing = NAMES - {p.name for p in DEST.glob("*.csv")}
    if missing:
        raise RuntimeError(f"Archivos faltantes en descarga: {missing}. Revisar ZIP UCI.")
    print("Datos listos:", ", ".join(sorted(NAMES)))


if __name__ == "__main__":
    main()
