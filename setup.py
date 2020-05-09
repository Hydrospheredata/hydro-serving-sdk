from os import path

from setuptools import setup, find_packages

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open("version", 'r') as f:
    version = f.read()

pkgs = find_packages(exclude=['tests', 'tests.*'])
print("FOUND PKGS", pkgs)

reqs = [
    'hydro-serving-grpc~=2.2.1',
    'sseclient-py~=1.7',
    'numpy~=1.18.3',
    'pyyaml~=5.3.1',
    'requests~=2.23.0',
    'requests_toolbelt~=0.9.1'
]

test_reqs = ['pytest~=5.4.1', 'requests_mock>=1.7.0']

setup(
    name='hydrosdk',
    version=version,
    description="Hydro-serving SDK",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://hydrosphere.io/",
    license="Apache 2.0",
    packages=pkgs,
    install_requires=reqs,
    include_package_data=True,
    setup_requires=['pytest-runner'],
    test_suite='tests',
    tests_require=test_reqs
)
