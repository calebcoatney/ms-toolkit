from pathlib import Path
from setuptools import setup, find_packages

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="ms-toolkit-nrel",
    version="0.1.0",
    description="Tools for mass spectrometry data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    license_files=("LICENSE",),
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "joblib",
        "gensim",
        "scikit-learn",
    ],
    python_requires=">=3.8",
)
