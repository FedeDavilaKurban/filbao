"""
# This script extracts code cells from a Jupyter notebook (.ipynb) file
# and writes them to a Python file (.py).
# It can be run from the command line or imported as a module.
# The output file will have the same name as the notebook but with a .py extension.
# If an output file is specified, it will be used instead.
# The script handles both reading the notebook and writing the code cells to a file.
# It prints the path of the output file after writing.
# The script is designed to be simple and efficient for extracting code from Jupyter notebooks.
# It assumes the notebook is well-formed and contains code cells.
# The script can be extended or modified to include additional features, such as error handling or logging
"""

import json
import sys
from pathlib import Path

def extract_code_cells(notebook_path, output_path=None):
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Extract code cells
    code_cells = []
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            code = ''.join(cell.get('source', []))
            code_cells.append(code + '\n')

    # Join all code cells
    full_code = '\n'.join(code_cells)

    # Determine output path
    if output_path is None:
        output_path = Path(notebook_path).with_suffix('.py')

    # Write to .py file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_code)

    print(f"Code cells written to {output_path}")

# Example usage:
# extract_code_cells("your_notebook.ipynb")
# Or from command line:
# python extract_ipynb_code.py your_notebook.ipynb [optional_output.py]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_ipynb_code.py <notebook.ipynb> [output.py]")
    else:
        notebook_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        extract_code_cells(notebook_file, output_file)
