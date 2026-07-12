"""Compatibility metadata for the macOS system Python's older setuptools."""

from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).parent

setup(
    name="ralphton-icml-reviewer",
    version="0.1.0",
    description="Evidence-grounded author/reviewer agent team and convergence loop",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=["ralphton_icml"],
    py_modules=["prompts", "review_prompts"],
    data_files=[(".", ["reviewer_instruction.md"])],
    python_requires=">=3.9",
    entry_points={"console_scripts": ["ralphton-icml=ralphton_icml.cli:main"]},
)
