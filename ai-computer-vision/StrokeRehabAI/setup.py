"""
setup.py
========
Packaging metadata for StrokeRehabAI. Enables editable installs
(`pip install -e .`) so the project's packages (camera, models,
training, etc.) are importable from anywhere without manual PYTHONPATH
manipulation.
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="strokerehab-ai",
    version="0.1.0",
    description="AI-Assisted Stroke Rehabilitation Exercise Assessment System using Computer Vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="StrokeRehabAI Project",
    python_requires=">=3.10,<3.11",
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    install_requires=[
        "torch>=2.3.1",
        "torchvision>=0.18.1",
        "torchmetrics>=1.4.0",
        "opencv-python>=4.10.0",
        "mediapipe==0.10.14",  # pinned exactly: newer releases removed the
                                 # mp.solutions.pose API this project uses —
                                 # see docs/installation_guide.md
        "numpy>=1.26.4",
        "pandas>=2.2.2",
        "scikit-learn>=1.5.1",
        "matplotlib>=3.9.1",
        "plotly>=5.23.0",
        "PyYAML>=6.0.1",
        "rich>=13.7.1",
        "onnxruntime-gpu>=1.18.1",
        "onnx>=1.16.1",
        "streamlit>=1.37.0",
        "tensorboard>=2.17.0",
        "tqdm>=4.66.4",
    ],
    extras_require={
        "dev": ["pytest>=8.3.2", "pytest-cov>=5.0.0"],
    },
    entry_points={
        "console_scripts": [
            "strokerehab-train=training.trainer:main",
            "strokerehab-infer=inference.realtime_pipeline:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 3 - Alpha",
    ],
)
