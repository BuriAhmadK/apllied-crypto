from setuptools import setup, find_packages

setup(
    name="fhe-medical-demo",
    version="1.0.0",
    description="FHE Medical Research Demonstration with Performance Comparison",
    packages=find_packages(),
    install_requires=[
        "tenseal>=0.3.0",
        "phe>=1.4.0", 
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "memory-profiler>=0.60.0",
        "psutil>=5.9.0"
    ],
    python_requires=">=3.8",
)