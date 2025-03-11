from setuptools import setup


setup(
    name="wandb-utils",
    version=0.1,
    description="Utilities for interacting with Weights and Biases and mlflow",
    zip_safe=False,  
    packages=["wandb_utils"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
    ],
    install_requires=[
        "mlflow",
        "wandb"
    ]
)
