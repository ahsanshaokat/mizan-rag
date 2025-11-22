from pathlib import Path

def load_text_file(path: str, encoding: str = "utf-8") -> str:
    return Path(path).read_text(encoding=encoding)
