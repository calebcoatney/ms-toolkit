from setuptools import setup, find_packages

setup(
    name="ms-toolkit",
    version="0.1.0",
    description="Tools for mass spectrometry data analysis",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "joblib",
        "gensim",
        "scikit-learn",
    ],
    python_requires=">=3.8",
)