import json
from pathlib import Path

out = Path(r"C:\Users\benja\AppData\Local\Temp\claude\C--Users-benja-OneDrive-Escritorio-Fraud-Detection\8dfc0995-f3c2-4aaa-b090-f5c18f3653fd\tasks\w5ak8bitn.output")
d = json.loads(out.read_text(encoding="utf-8"))
target = Path(r"C:\Users\benja\OneDrive\Escritorio\CAAD PROYECT\dron_pose\augmentar_dataset.py")
target.write_text(d["result"]["script"], encoding="utf-8", newline="\n")
print("written", len(d["result"]["script"]), "chars to", target)
