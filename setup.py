from setuptools import setup, find_packages

with open("requirements.txt", 'r') as f:
    REQUIREMENTS = f.read()

with open("README.md") as f:
    README = f.readlines()

with open("LICENSE.apache") as f:
    LICENSE = f.readlines()

LICENSE = LICENSE + ['\n']

with open("LICENSE.gpl") as f:
    LICENSE = LICENSE + f.readlines()

setup(name="look-a-like", packages=find_packages(exclude="tests.*.*.*"),
      install_requires=REQUIREMENTS, version="0.1.0",
      extras_require={"spark": ["pyspark=2.4.3", "pandas=0.24.2"]},
      author="Edward Turner",
      author_email="edward.turnerr@gmail.com",
      description=README,
      license=LICENSE,
      python_requires='==Python3.7'
      )
