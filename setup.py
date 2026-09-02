from setuptools import setup, find_packages

setup(
    name="multimodal-nfm-net",
    version="2.1.0",
    packages=find_packages(),
    py_modules=["train", "pretrain", "train_omni"],
    entry_points={
        "console_scripts": [
            "nfm-train=train:main",
            "nfm-pretrain=pretrain:main",
        ],
    },
)
