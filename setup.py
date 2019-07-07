from setuptools import setup, find_packages

with open("requirements.txt", 'r') as f:
    REQUIREMENTS = f.read()

with open("README.md") as f:
    README = f.readlines()

LICENSE = ''

setup(name="look-a-like", packages=find_packages(exclude=("lal.spark.*",)),
      install_requires=REQUIREMENTS, version="0.0.0",
      author="Edward Turner",
      author_email="edward.turnerr@gmail.com",
      description=README,
      license=LICENSE,
      python_requires='==Python3.7'
      )
