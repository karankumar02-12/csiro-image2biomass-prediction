# ----- setup.py -----
from setuptools import setup, find_packages

setup(
    name="csiro-biomass",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "opencv-python>=4.8.0",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.6",
        "lightgbm>=4.0.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "tqdm>=4.65.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="CSIRO Image2Biomass Prediction Solution",
    keywords="computer-vision machine-learning agriculture biomass",
    python_requires=">=3.8",
)
