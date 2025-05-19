from setuptools import setup, find_packages

setup(
    name="asl_translator",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-multipart",
        "aiofiles"
    ],
)
