import ast
import json
import re
import sys
import zipfile
from pathlib import Path


zip_path = Path(sys.argv[1])
patterns = {
    "network": re.compile(r"\b(requests|urllib|httpx|aiohttp|socket|ftplib)\b|https?://", re.I),
    "process": re.compile(r"\b(subprocess|os\.system|Popen|spawn|execv|ShellExecute)\b", re.I),
    "dynamic_exec": re.compile(r"\b(eval|exec|compile)\s*\(", re.I),
    "destructive": re.compile(r"\b(rmtree|unlink|remove|del /|rm -rf)\b", re.I),
}
findings = []
parse_errors = []
binary_findings = []
model_extensions = {
    ".pkl", ".pickle", ".joblib", ".pt", ".pth", ".cbm", ".json", ".ubj", ".txt", ".bin"
}

with zipfile.ZipFile(zip_path) as archive:
    bad_entry = archive.testzip()
    names = archive.namelist()
    for name in names:
        lower_name = name.lower()
        if lower_name.endswith(".py"):
            raw = archive.read(name)
            source = raw.decode("utf-8", errors="replace")
            try:
                ast.parse(source, filename=name)
            except SyntaxError as error:
                parse_errors.append({"file": name, "line": error.lineno, "message": error.msg})
            for line_number, line in enumerate(source.splitlines(), 1):
                for kind, regex in patterns.items():
                    if regex.search(line):
                        findings.append(
                            {"kind": kind, "file": name, "line": line_number, "text": line.strip()[:300]}
                        )
        elif Path(lower_name).suffix in model_extensions:
            raw_lower = archive.read(name).lower()
            for token in (
                b"os\nsystem",
                b"posix\nsystem",
                b"nt\nsystem",
                b"subprocess",
                b"builtins\neval",
                b"builtins\nexec",
                b"__reduce__",
                b"requests",
                b"http://",
                b"https://",
            ):
                if token.lower() in raw_lower:
                    binary_findings.append({"file": name, "token": token.decode("latin1")})
                    break

report = {
    "zip": str(zip_path),
    "entry_count": len(names),
    "crc_bad_entry": bad_entry,
    "python_count": sum(name.lower().endswith(".py") for name in names),
    "python_parse_errors": parse_errors,
    "text_findings": findings,
    "binary_suspicious_strings": binary_findings,
}
print(json.dumps(report, ensure_ascii=False, indent=2))
