import json

with open("notebooks/02_zero_shot_and_sentiment.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("results/extracted_cell_13.txt", "w", encoding="utf-8") as out_f:
    for out in nb["cells"][13]["outputs"]:
        if out.get("output_type") == "stream":
            out_f.write("".join(out.get("text", [])))
