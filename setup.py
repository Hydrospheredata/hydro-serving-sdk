from setuptools import setup, find_packages

pkgs = find_packages(exclude=['tests', 'tests.*'])
print("FOUND PKGS", pkgs)

reqs = [
    'hydro-serving-grpc==2.1.0',
    'sseclient-py~=1.7',
    'numpy',
    'pyyaml~=5.1.2',
    'requests~=2.22.0',
    'requests_toolbelt~=0.9.1'
]

test_reqs = ['pytest>=3.8.0', 'requests_mock>=1.7.0']

setup(
    name='hydrosdk',
    version='2.2.0-dev',
    description="Hydro-serving SDK",
    url="https://hydrosphere.io/",
    license="Apache 2.0",
    packages=pkgs,
    install_requires=reqs,
    include_package_data=True,
    setup_requires=['pytest-runner'],
    test_suite='tests',
    tests_require=test_reqs
)
