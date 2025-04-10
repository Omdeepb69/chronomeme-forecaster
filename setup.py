import setuptools
import os

_PROJECT_NAME = "ChronoMeme Forecaster"
_PROJECT_VERSION = "0.1.0"
_PROJECT_AUTHOR = "Omdeep Borkar"
_PROJECT_AUTHOR_EMAIL = "omdeeborkar@gmail.com"
_PROJECT_URL = "https://github.com/Omdeepb69/ChronoMeme Forecaster"
_PROJECT_DESCRIPTION = (
    "Predicts the short-term 'virality' or trend score of internet memes "
    "based on social media mention frequency and sentiment analysis over time."
)

_REQUIREMENTS = [
    "pandas>=1.0.0",
    "numpy>=1.18.0",
    "statsmodels>=0.11.0",
    "prophet>=1.0", # Note: Prophet installation can be complex, consider conda or specific instructions
    "nltk>=3.5",
    "vaderSentiment>=3.3.2",
    "matplotlib>=3.2.0",
    "seaborn>=0.10.0",
    "scikit-learn>=0.22.0",
]

_LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
_README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")

try:
    with open(_README_PATH, "r", encoding="utf-8") as f:
        _LONG_DESCRIPTION = f.read()
except FileNotFoundError:
    _LONG_DESCRIPTION = _PROJECT_DESCRIPTION


setuptools.setup(
    name=_PROJECT_NAME,
    version=_PROJECT_VERSION,
    author=_PROJECT_AUTHOR,
    author_email=_PROJECT_AUTHOR_EMAIL,
    description=_PROJECT_DESCRIPTION,
    long_description=_LONG_DESCRIPTION,
    long_description_content_type=_LONG_DESCRIPTION_CONTENT_TYPE,
    url=_PROJECT_URL,
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=_REQUIREMENTS,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License", # Assuming MIT, change if needed
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Sociology",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    entry_points={
        'console_scripts': [
            # Add command-line scripts here if needed
            # 'chronomeme=chronomeme_forecaster.cli:main',
        ],
    },
    project_urls={
        "Bug Tracker": f"{_PROJECT_URL}/issues",
        "Source Code": _PROJECT_URL,
    },
)