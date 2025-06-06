"""
Setup script for Cerebrovascular Aneurysm Analysis project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aneurysm-analysis",
    version="0.1.0",
    author="KAIST URP Team",
    author_email="your-email@kaist.ac.kr",
    description="Cerebrovascular aneurysm analysis using PyAnsys and machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/urp-2025",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "pyvista>=0.42.0",
        "trimesh>=3.23.0",
        "matplotlib>=3.7.0",
        "plotly>=5.17.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "preprocess-meshes=rupture_risk_prediction.src.preprocessing.prepare_meshes:main",
            "run-stress-analysis=rupture_risk_prediction.src.fea_analysis.stress_analysis:main",
        ],
    },
) 