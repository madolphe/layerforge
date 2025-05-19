from setuptools import setup, find_packages

setup(
    name="layerforge",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21.0,<2.0.0",
        "PyYAML>=6.0.2",
        "scipy>=1.15.3",
        "tmm-fast",
        "torch>=2.2.0",
        "matplotlib>=3.5.0",
        "requests",
        "refractiveindex @ git+https://github.com/toftul/refractiveindex.git"
    ],
) 