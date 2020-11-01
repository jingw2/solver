try:
    from setuptools import setup
except:
    from distutils.core import setup
from setuptools import find_packages

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# 必须装的环境
with open(path.join(this_directory, "requirements.txt")) as fp:
    install_requires = fp.read().strip().split("\n")

VERSION = "0.0.2" # 每次更新的版本号需要不同，PyPI不支持覆盖
LICENSE = 'MIT'
setup(
      version=VERSION,
      setup_requires=["numpy"],
      install_requires=install_requires,
      name='forecastability',
      description='forecastability analysis',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/jingw2/solver/tree/master/forecastability',
      author='Jing Wang',
      author_email='jingw2@foxmail.com',
      license=LICENSE,
      packages=find_packages(),
      python_requires='>=3.6')
