"""
setup.py
Setup file for the Driver Drowsiness Detection System.
Makes the project installable as a Python package.
"""

from setuptools import setup, find_packages

setup(
    name="driver-drowsiness-detection",
    version="1.0.0",
    description="Real-time driver drowsiness detection using MediaPipe Face Mesh",
    author="Subhan",
    author_email="subhan@uajk.edu.pk",
    url="https://github.com/husnainzahid050-hub/cv-driver-drowness-detection",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pygame>=2.5.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
    },
    entry_points={
        "console_scripts": [
            "drowsiness-detect=main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
