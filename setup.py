# Copyright 2020 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from setuptools import setup, find_packages


# Load package info, without importing the package
basedir = os.path.dirname(os.path.abspath(__file__))
package_info_path = os.path.join(basedir, "dwave", "plugins", "qiskit", "package_info.py")
package_info = {}
with open(package_info_path, encoding='utf-8') as f:
    exec(f.read(), package_info)


install_requires = ['qiskit-aqua>=0.8.0', 'dwave-system>=1.2.0']

python_requires = ">=3.6"

classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
]

packages = [pkg for pkg in find_packages() if pkg.startswith('dwave.plugins.qiskit')]

setup(
    name=package_info['__package_name__'],
    version=package_info['__version__'],
    author=package_info['__author__'],
    author_email=package_info['__author_email__'],
    description=package_info['__description__'],
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url=package_info['__url__'],
    license=package_info['__license__'],
    packages=packages,
    install_requires=install_requires,
    python_requires=python_requires,
    classifiers=classifiers,
    zip_safe=False,
)
