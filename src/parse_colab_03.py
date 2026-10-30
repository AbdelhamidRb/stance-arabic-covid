import json

with open("notebooks/03_deep_learning_scratch.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("results/extracted_notebook_03_outputs.txt", "w", encoding="utf-8") as out_f:
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            outputs = cell.get("outputs", [])
            if outputs:
                out_f.write(f"\n==================== CODE CELL {i} ====================\n")
                for out in outputs:
                    if out.get("output_type") == "stream":
                        text = "".join(out.get("text", []))
                        out_f.write(text)
                    elif "data" in out and "text/plain" in out["data"]:
                        text = "".join(out["data"]["text/plain"])
                        out_f.write(text + "\n")

print("Succès extraction complète sans crash d'encodage !")
