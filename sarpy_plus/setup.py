from setuptools import setup, find_packages

setup(
    name="sarpy-plus",
    version="0.1.0",
    description="Modular Python SAR simulation and processing library (with JAX)",
    author="Adam Cohen",
    packages=find_packages(),
    install_requires=[
        "jax>=0.4.20",
        "jaxlib>=0.4.20",
        "matplotlib>=3.5.0",
    ],
    python_requires=">=3.8",
    install_optional=[
        "sarpy >= 1.3.61"
    ],
)