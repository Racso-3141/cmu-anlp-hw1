from setuptools import find_packages, setup

setup(
    name="anlp-f2026-hw1",
    version="2026.1.0",
    description="CMU 11-711 Advanced NLP, Fall 2026 — Homework 1",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
)
