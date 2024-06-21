#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

setup(
    python_requires=">=3.11",
    description="API for sparam plotting web app",
    install_requires=[
        "fastapi",
        "numpy",
        "scipy",
        "uvicorn",
        "xarray",
        "rich_click",
    ],
    include_package_data=True,
    keywords="splot_api",
    name="frf-splot_api",
    packages=find_packages(
        include=[
            "splot_api",
            "splot_api.*",
        ]
    ),
    entry_points={
        "console_scripts": [
            "splot_api = splot_api.cli:cli",
        ],
    },
    version="0.1.0",
    zip_safe=False,
)
