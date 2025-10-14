import os
import re
import shutil
import zipfile
import subprocess
from typing import List
from pathlib import Path

import nbformat

ASSIGNMENTS_FOLDER = "src/assignments"
NOTEBOOK_FILES = [
    "knn_part_1.ipynb",
    "knn_part_2.ipynb",
    "linear_part_1.ipynb",
    "linear_part_2.ipynb",
    "mlp_part_1.ipynb",
    "mlp_part_2.ipynb",
]

scripts_dir = os.path.join(os.path.dirname(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(scripts_dir, os.pardir, os.pardir))

ZIP_FILE = "hw1.zip"


def preprocess_notebooks(
    input_files: List[Path], output_files: List[Path], root_path: Path
) -> None:
    """Preprocess Jupyter notebooks by replacing relative image paths with absolute paths.
    
    If preprocessing fails, the original notebook will still be copied to the output folder.
    
    Args:
        input_files (List[Path]): List of input Jupyter notebook files.
        output_files (List[Path]): List of output Jupyter notebook files.
        root_path (Path): Path to the root folder.
    
    Returns:
        None
    """
    for input_path, output_path in zip(input_files, output_files):
        try:
            # Attempt to preprocess the notebook
            with open(input_path, "r") as infile:
                notebook = nbformat.read(infile, as_version=4)

            # Replace relative paths in markdown cells
            for cell in notebook.cells:
                if cell.cell_type == "markdown":
                    cell.source = cell.source.replace(
                        '<img src="', f'<img src="{root_path}/'
                    )

                    # Handle markdown images: ![Alt](path)
                    cell.source = re.sub(
                        r'!\[([^\]]*)\]\((?!http)([^)]+)\)',
                        lambda m: f'![{m.group(1)}]({root_path}/{m.group(2)})',
                        cell.source
                    )

            # Write the preprocessed notebook to the output file
            with open(output_path, "w") as outfile:
                nbformat.write(notebook, outfile)

            print(f"\tINFO: Preprocessed {input_path.name}")
        
        except Exception as e:
            # If preprocessing fails, copy the original notebook to the output location
            shutil.copy(input_path, output_path)
            print(f"\tWARNING: Preprocessing failed for {input_path.name}. Copied without modification. Error: {e}")



def create_html_files(jupyter_files: List[Path], remove_original: bool = True):
    """This function creates HTML files from Jupyter notebooks.

    Args:
        jupyter_files (list): List of Jupyter notebook files.
        remove_original (bool, optional): Whether to remove the original Jupyter notebook files. Defaults to True.

    Returns:
        List[str]: List of HTML files.

    """

    for f in jupyter_files:
        subprocess.run(["jupyter", "nbconvert", "--to", "html", "--embed-images", f])
        print(f"\tINFO: Created {os.path.basename(f).replace('.ipynb', '.html')}")

    if remove_original:
        print(f"\tINFO: Removing original Jupyter notebook files...")
        for f in jupyter_files:
            print(f"\tINFO: Removed {f}")
            os.remove(f)


def submit():
    project_dir = Path(PROJECT_DIR)
    output_file = project_dir / ZIP_FILE

    notebooks_folder = project_dir / "notebooks"
    assignments_folder = project_dir / "src" / "assignments"

    notebook_files = [project_dir / f for f in NOTEBOOK_FILES]
    preprocessed_notebook_files = [notebooks_folder / f for f in NOTEBOOK_FILES]

    # Setup: Create the notebooks folder
    print(f"\n================= SETUP =================\n")
    if not notebooks_folder.exists():
        notebooks_folder.mkdir(parents=True, exist_ok=True)
        print(f"\tINFO: Created {notebooks_folder}.")
    else:
        print(f"\tERROR: {notebooks_folder} already exists. Exiting...")
        exit(1)

    # Preprocess the notebooks
    print(f"\n================= PREPROCESSING =================\n")
    preprocess_notebooks(notebook_files, preprocessed_notebook_files, project_dir)

    # Create the HTML files
    print(f"\n================= CREATING HTML FILES =================\n")
    create_html_files(preprocessed_notebook_files, remove_original=False)

    # Zip the assignments and notebooks folders
    print(f"\n================= CREATING ZIP FILE =================\n")
    create_zip_file(output_file, [assignments_folder, notebooks_folder], project_dir)

    # Cleanup: Remove the notebooks folder
    print(f"\n================= CLEANUP =================\n")
    if notebooks_folder.exists():
        shutil.rmtree(notebooks_folder)
        print(f"\tINFO: Removed {notebooks_folder}")
        
def create_zip_file(output_zip_path: Path, folders_to_zip: List[Path], project_dir: Path) -> None:
    with zipfile.ZipFile(output_zip_path, 'w') as zipf:
        for folder in folders_to_zip:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(project_dir)
                    zipf.write(file_path, arcname)
    print(f"\tINFO: Created zip file at {output_zip_path}")


if __name__ == "__main__":
    submit()
