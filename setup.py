from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import sys

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        # Download NLTK data
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('stopwords', quiet=True)
            print("NLTK data downloaded successfully")
        except Exception as e:
            print(f"Warning: Could not download NLTK data: {e}")

setup(
    name="careerpathway",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.60.0",
        "pyyaml>=6.0",
        "requests>=2.25.0",
        "nltk>=3.6",
        "nrclex>=3.0.0",
        "openai>=1.0.0",
        "anthropic>=0.3.0",
        "google-generativeai>=0.3.0",
        "vllm>=0.2.0",
        "fire>=0.4.0",
        "datasets>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "jupyter>=1.0",
        ],
        "gpu": [
            "torch[cuda]>=1.9.0",
        ]
    },
    cmdclass={
        'install': PostInstallCommand,
    },
    author="Seungbeen",
    description="A package for career pathway analysis using LLMs",
    long_description="A package for career pathway analysis using LLMs",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/careerpathway",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
