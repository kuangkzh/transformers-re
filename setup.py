from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='transformers-re',
    version='1.1.4',
    author='kuangzh',
    author_email='1432245553@qq.com',
    url='https://github.com/kuangkzh/transformers-re',
    description='A Regular Expression constraint for Language Models of transformers. With this module, you can force '
                'the LLMs to generate following your regex. Using regex in tokens and tensors are also implemented in '
                'this project.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    py_modules=['transformers_re'],
    install_requires=[
        "multiprocess",
        "transformers>=4.0.0",
        "regex>=2014.04.10"
    ],
)
