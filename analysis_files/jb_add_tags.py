import nbformat as nbf
from glob import glob

# Collect a list of all notebooks in the content folder
notebooks = glob("./analysis_files/*.ipynb", recursive=True)
print(f"Notebooks being ammended: {notebooks}")
# Search through each notebook and set "hide_input": true for all code cells
for ipath in notebooks:
    ntbk = nbf.read(ipath, nbf.NO_CONVERT)

    for cell in ntbk.cells:
        if cell["cell_type"] == "code":
            cell["metadata"].pop("hide_input", None)  # remove the old syntax
            if "tags" in cell["metadata"]:
                if "hide-input" not in cell["metadata"]["tags"]:
                    cell["metadata"]["tags"].append("hide-input")
            else:
                cell["metadata"]["tags"] = ["hide-input"]

    nbf.write(ntbk, ipath)
