from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='transformers-re',
    version='0.0.6',
    author='kuangzh',
    author_email='1432245553@qq.com',
    url='https://github.com/kuangkzh/kzhutil',
    description='RegEx constraint for Language Model output',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "multiprocess",
        "transformers>=4.0.0",
        "regex>=2014.04.10"
    ],
)
