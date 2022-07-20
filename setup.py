import setuptools
import sys

if sys.version_info < (3,7,13):
    sys.exit('Sorry, Python < 3.7.13 is not supported')

setuptools.setup(
    name='relex',
    author='Ali NCIBI',
    description='An easy to use PyTorch-based library for state-of-the-art semantic relations extraction.',
    url='https://github.com/ylaxor/relex',
    python_requires='>=3.7.13',
    install_requires=['flair', 'torchmetrics', 'pandas', 'numpy'],
    packages=setuptools.find_packages(exclude="tests"),
    license="MIT",
    long_description=open("./README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    version="0.1.1",
    download_url = 'https://github.com/ylaxor/relex/archive/refs/tags/v0.1.1.tar.gz',  
    keywords = ['PyTorch', 'Semantic Relation Extraction', 'Pipeline', 'Text analysis', 'NLP', 'NLU', 'Deep learning', 'AI'],  
)