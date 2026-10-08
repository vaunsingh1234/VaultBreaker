from setuptools import setup, find_packages

setup(
    name="vaultbreaker",
    version="1.0.0",
    description="Unified Multi-Modal Steganography Detection Pipeline",
    author="VaultBreaker Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
