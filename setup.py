from setuptools import setup, find_packages

setup(name="b4tf",
      author="Hiroyuki Yamada",
      version="0.0.0",
      description="Bayes for TensorFlow",
      install_requires=["tensorflow>=2","tensorflow_probability"],
      packages=find_packages(),
      classifiers = ["Development Status :: 3 - Alpha",
                     "Intended Audience :: Developers",
                     "Intended Audience :: Science/Research",
                     "License :: OSI Approved :: MIT License",
                     "Programming Language :: Python :: 3 :: Only",
                     "Topic :: Scientific/Engineering :: Artificial Intelligence"])
