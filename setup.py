import os
import warnings

from setuptools import setup, find_packages


description = "Bayian Neural Network for TensorFlow"
README = os.path.join(os.path.abspath(os.path.dirname(__file__)),'README.md')
if os.path.exists(README):
    with open(README,encoding='utf-8') as f:
        long_description = f.read()
    long_description_content_type='text/markdown'
else:
    warnings.warn("No README.md")
    long_description =  description
    long_description_content_type='text/plain'


setup(name="b4tf",
      author="Hiroyuki Yamada",
      version="0.0.6",
      description=description,
      install_requires=["tensorflow>=2","tensorflow_probability"],
      url="https://gitlab.com/ymd_h/b4tf",
      packages=find_packages("."),
      classifiers = ["Development Status :: 3 - Alpha",
                     "Intended Audience :: Developers",
                     "Intended Audience :: Science/Research",
                     "License :: OSI Approved :: MIT License",
                     "Programming Language :: Python :: 3 :: Only",
                     "Topic :: Scientific/Engineering :: Artificial Intelligence"])
