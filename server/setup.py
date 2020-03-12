import os

from setuptools import setup, find_packages
''
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, '../README.md')) as f:
    README = f.read()
with open(os.path.join(here, '../CHANGES.md')) as f:
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
    version='0.0',
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
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    extras_require={
        'testing': tests_require,
    },
    package_data={
        'kwola': [
            'tests/data/*.json',
        ]
    },
    install_requires=requires,
    entry_points={
        'paste.app_factory': [
            'main = kwola:main',
        ],
        'console_scripts': [
            'kwola_clear_db = kwola.bin.clear_db:main',
            'kwola_train_agent = kwola.bin.train_agent:main'
        ]
    },
)

