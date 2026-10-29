import json

with open("notebooks/02_zero_shot_and_sentiment.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("=== CELL OUTPUTS FROM COLAB ===")
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        outputs = cell.get("outputs", [])
        if outputs:
            print(f"\n--- Code Cell {i} ---")
            for out in outputs:
                if out.get("output_type") == "stream":
                    print("".join(out.get("text", [])))
                elif "data" in out and "text/plain" in out["data"]:
                    print("".join(out["data"]["text/plain"]))
