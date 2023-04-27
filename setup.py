#!/usr/bin/python
# coding: utf8
# /*##########################################################################
#
# Copyright (c) 2023 Hongyang Dong
#
# Most of functions are reproduced from nDTomo for CT image reconstruction
# https://github.com/antonyvam/nDTomo
#
# This package is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
# 
# This package is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>
# 
# On Debian systems, the complete text of the GNU General
# Public License version 3 can be found in "/usr/share/common-licenses/GPL-3".
#
# ###########################################################################*/
  

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
	name="sd2i",
	version="2023.04",
	description="Single Digit to Image reconstruction tool",
	url="https://github.com/robindong3/SD2I",
	author="H. Dong",
	author_email="robondong3@gmail.com",
	install_requires=[
		"h5py", "matplotlib", "numpy","scikit-image",  "xdesign", "cached_property", "tqdm", "scikit-learn"
	],
	packages=find_packages(),
    package_data={
        '': ['*.txt', '*.rst'],
    },
	license="LICENSE.txt",
	classifiers=[
		"Intended Audience :: Science/Research",
		"Topic :: Scientific/Engineering",
		"Topic :: Scientific/Engineering :: Chemistry",
		"Topic :: Scientific/Engineering :: Visualization",
		],
) 

