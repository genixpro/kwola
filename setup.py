#
#     Kwola is an AI algorithm that learns how to use other programs
#     automatically so that it can find bugs in them.
#
#     Copyright (C) 2020  Kwola Software Testing Inc.
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Affero General Public License as
#     published by the Free Software Foundation, either version 3 of the
#     License, or (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
#


import os

from setuptools import setup, find_packages
''
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md')) as f:
    README = f.read()
with open(os.path.join(here, 'CHANGES.md')) as f:
    CHANGES = f.read()
with open('requirements.txt', 'rt') as f:
    requires = f.readlines()

tests_require = [
    'WebTest >= 1.3.1',  # py3 compat
    'pytest >= 3.7.4',
    'pytest-cov',
]

setup(
    name='kwola',
    version='0.0.1',
    description='Kwola makes an AI powered tooling for finding bugs in software',
    long_description=README + '\n\n' + CHANGES,
    classifiers=[
        'Programming Language :: Python',
        'Framework :: Pyramid',
        'Topic :: Internet :: WWW/HTTP',
        'Topic :: Internet :: WWW/HTTP :: WSGI :: Application',
    ],
    author='',
    author_email='',
    url='',
    keywords='web pyramid pylons',
    packages=['kwola'],
    include_package_data=True,
    zip_safe=False,
    extras_require={
        'testing': tests_require,
    },
    package_data={
        'kwola': [
            'tests/data/*.json',
            'config/prebuilt_configs/*.json'
        ]
    },
    install_requires=requires,
    entry_points={
        'paste.app_factory': [
            'main = kwola:main',
        ],
        'console_scripts': [
            'kwola= kwola.bin.main:main',
            'kwola_clear_db = kwola.bin.clear_db:main',
            'kwola_train_agent = kwola.bin.train_agent:main',
            'kwola_run_test_sequence = kwola.bin.run_test_sequence:main'
        ]
    },
)

