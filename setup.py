from setuptools import setup, find_packages

setup(name="b4tf",
      author="Hiroyuki Yamada",
      version="0.0.0",
      description="Bayes for TensorFlow",
      install_requires=["tensorflow>=2","tensorflow_probability"],
      packages=find_packages())
