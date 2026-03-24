import os, re
pdir = "report_type_plugins"
for d in sorted(os.listdir(pdir)):
    pp = os.path.join(pdir, d, "plugin.py")
    if not os.path.isfile(pp):
        continue
    src = open(pp, encoding="utf-8").read()
    uses_shared = "_twamp_shared" in src
    has_mp = "metrics_profile" in src
    has_old = bool(re.search(r'"id"\s*:', src)) and not has_mp
    lines = len(src.splitlines())
    print(f"{d:55s} lines={lines:4d} shared={uses_shared} mp={has_mp} old={has_old}")
