from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

with open("version", 'r') as f:
    version = f.read()

pkgs = find_packages(exclude=['tests', 'tests.*'])
print("FOUND PKGS", pkgs)

reqs = [
    'importlib_metadata~=1.7.0',
    'hydro-serving-grpc~=2.4.0',
    'sseclient-py~=1.7',
    'numpy~=1.18.3',
    'pandas~=1.0.3',
    'pyyaml~=5.3.1',
    'requests~=2.23.0',
    "dataclasses==0.7;python_version<'3.7'",
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
    classifiers=["License :: OSI Approved :: Apache Software License",
                 "Intended Audience :: Developers",
                 "Intended Audience :: Information Technology",
                 "Topic :: Software Development :: Libraries :: Python Modules"],
    install_requires=reqs,
    include_package_data=True,
    python_requires=">=3.6",
    setup_requires=['pytest-runner'],
    test_suite='tests',
    tests_require=test_reqs
)
