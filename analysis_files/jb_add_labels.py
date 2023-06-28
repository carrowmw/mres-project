import os
import nbformat as nbf
from glob import glob


def add_labels(nb):
    """
    Adds unique labels to each header in the given Jupyter notebook object.
    If a label already exists for a header, it is replaced.

    Parameters
    ----------
    nb : nbf.NotebookNode
        The Jupyter notebook object.

    Returns
    -------
    nb : nbf.NotebookNode
        The updated Jupyter notebook object with added labels.
    """
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            lines = cell.source.split("\n")
            new_lines = []
            for i, line in enumerate(lines):
                stripped_line = line.strip()
                if stripped_line.startswith("#"):
                    # determine header level
                    level = stripped_line.count(
                        "#", 0, len(stripped_line) - len(stripped_line.lstrip("#"))
                    )
                    # Extract number part
                    title = stripped_line.lstrip("# ").strip()
                    # Check if the first non-space character sequence is a number (potentially fractional)
                    number_part = title.split(" ", 1)[0]
                    if any(char.isdigit() for char in number_part):
                        # Check if previous line is a label and remove it
                        if (
                            i > 0
                            and lines[i - 1].strip().startswith("(")
                            and lines[i - 1].strip().endswith(")=")
                        ):
                            new_lines.pop()
                        # Prepend label
                        new_lines.append(f"({number_part})=")
                        new_lines.append(f'{"#" * level} {title}')
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            cell.source = "\n".join(new_lines)
    return nb


def process_notebooks(directory):
    """
    Processes all Jupyter notebooks in the specified directory by adding unique labels to each header.

    Parameters
    ----------
    directory : str
        The path to the directory containing the Jupyter notebooks.
    """
    notebooks = glob(os.path.join(directory, "*.ipynb"))
    for notebook in notebooks:
        with open(notebook, "r") as f:
            nb = nbf.read(f, as_version=4)
        nb = add_labels(nb)
        with open(notebook, "w") as f:
            nbf.write(nb, f)


# Process all notebooks in 'analysis_files' directory
process_notebooks(r"C:\#code\#python\#current\mres-project\analysis_files")
