import json

with open("notebooks/03_deep_learning_scratch.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        has_output = len(cell.get("outputs", [])) > 0
        src_preview = "".join(cell.get("source", []))[:60].replace("\n", " ")
        print(f"Cell {i} (code): has_output={has_output} | {src_preview}")
