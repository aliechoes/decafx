# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='decafx',
      version='0.6.4',
      description='Deep CAE feature extraction',
      url='http://github.com/nchlis',
      author='Nikolaos Kosmas Chlis',
      author_email='nchlis@isc.tuc.gr',
      license='GNU GPLv3',
      packages=['decafx'],
               install_requires=[
               'numpy',
               'scikit-image',
               'keras',
      ])
#      ,zip_safe=False)