from setuptools import setup, find_packages

setup(
    name = 'image_similarity',
    version = '1.0',    
    description = 'A better image dataset loader for TensorFlow.',
    url = 'https://github.com/UM2ii/image_similarity',
    author = 'Pranav Kulkarni',
    author_email = 'pranavkop@live.com',
    license = 'Apache License',
    packages = find_packages(),
    install_requires = [
      'transformers>=4.6.0,<5.0.0',
      'tqdm',
      'torch>=1.6.0',
      'torchvision',
      'numpy',
      'scikit-learn',
      'scipy',
      'nltk',
      'sentencepiece',
      'huggingface-hub>=0.4.0'
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires = ">=3.6",
)