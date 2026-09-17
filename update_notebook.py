import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
notebook_path = ROOT / "analysis.ipynb"

print(f"Reading notebook from: {notebook_path}")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Create setup cells
setup_md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Environment Setup & Dependency Installation\n",
        "\n",
        "Run the cell below to install all required Python modules for running this notebook."
    ]
}

pip_install_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Install all required Python modules for this notebook\n",
        "%pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib\n"
    ]
}

# Check if setup cells already inserted
has_pip_cell = False
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        source_str = "".join(cell.get("source", []))
        if "%pip install" in source_str or "!pip install" in source_str:
            has_pip_cell = True
            break

if not has_pip_cell:
    # Insert cells right after title markdown cell (cell index 1)
    nb["cells"].insert(1, setup_md_cell)
    nb["cells"].insert(2, pip_install_cell)

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print("analysis.ipynb updated successfully!")
else:
    print("Pip install cell already present in analysis.ipynb!")
