"""lab/eda_stats.json + exp/24_dashboard.tmpl.html -> lab/eda_dashboard.html"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
t = open(os.path.join(ROOT, "exp", "24_dashboard.tmpl.html"), encoding="utf-8").read()
d = json.load(open(os.path.join(ROOT, "lab", "eda_stats.json"), encoding="utf-8"))
s = json.dumps(d, ensure_ascii=False, separators=(",", ":"))
assert "</script" not in s and "__DATA__" in t
out = os.path.join(ROOT, "lab", "eda_dashboard.html")
open(out, "w", encoding="utf-8").write(t.replace("__DATA__", s))
print(f"{out}  {os.path.getsize(out)/1e3:.0f} KB")
