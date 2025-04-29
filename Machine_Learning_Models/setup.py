from setuptools import setup, find_packages

setup(
    name="Machine_Learning_Models",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "tensorflow",
        "scikit-learn",
        "matplotlib",
        "seaborn", 
        "joblib",
        "requests",
        "yfinance",
    ],
    entry_points={
        "console_scripts": [
            "nasdaq_predict=Machine_Learning_Models.scripts.run_prediction:main",
        ],
    },
)