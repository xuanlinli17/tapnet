from setuptools import setup, find_packages


def read_requirements():
    with open("requirements_inference.txt", "r") as f:
        lines = [l.strip() for l in f.readlines()]
    install_requires = list(filter(None, lines))
    return install_requires


if __name__ == "__main__":
    setup(
        name='tapnet',
        version='0.0.1',
        description="TAPIR DeepMind",
        install_requires=read_requirements(),
        packages=find_packages(include=["tapnet*"]),
    )
