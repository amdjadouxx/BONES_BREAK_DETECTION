from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="pediatric-fracture-detection",
    version="1.0.0",
    author="AHMOD ALI",
    author_email="amdjadahmodali974@gmail.com",
    description="Détection automatique de fractures osseuses chez les enfants avec YOLOv8",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amdjadouxx/pediatric-fracture-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "black>=22.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "gradcam": [
            "grad-cam>=1.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fracture-detect=inference.predict:main",
            "fracture-webapp=webapp.app:main",
        ],
    },
)
