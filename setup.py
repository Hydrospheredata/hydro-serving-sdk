from setuptools import setup, find_packages

with open("version", 'r') as v:
    version = v.read().strip()

pkgs = find_packages(exclude=['tests', 'tests.*'])
print("FOUND PKGS", pkgs)

setup(
    name='hydrosdk',
    version=version,
    description="Hydro-serving SDK",
    author="Bulat Lutfullin",
    author_email='blutfullin@hydrosphere.io',
    url="https://hydrosphere.io/",
    license="Apache 2.0",
    packages=pkgs,
    install_requires=["hs~=2.0.6"],
    include_package_data=True,
    setup_requires=[
        'pytest-runner'
    ],
    test_suite='tests',
    tests_require=[
        'pytest>=3.8.0', 'requests_mock>=1.5.0', 'mock>=2.0.0'
    ]
)
