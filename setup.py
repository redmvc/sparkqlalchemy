from setuptools import find_namespace_packages, setup

setup(
    name="sparkqlalchemy",
    version="0.2.1",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    install_requires=[
        "setuptools",
        "pandas",
        "sqlalchemy",
        "typing_extensions",
    ],
)
