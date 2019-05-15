from setuptools import setup

with open("requirements.txt", 'r') as f:
    REQUIREMENTS = f.read()

with open("README.md") as f:
    README = f.readlines()

LICENSE = ''

setup(name="look-a-like", packages=['lal', ],
      install_requires=REQUIREMENTS, version="0.0.0",
      author="Edward Turner",
      author_email="edward.turnerr@gmail.com",
      description=README,
      license=LICENSE,
      python_requires='==Python3.7'
      )
