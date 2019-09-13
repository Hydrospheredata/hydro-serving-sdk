from setuptools import setup, find_packages

pkgs = find_packages(exclude=['tests', 'tests.*'])
print("FOUND PKGS", pkgs)

setup(
    name='hydrosdk',
    version='2.1.0-dev0',
    description="Hydro-serving SDK",
    author="Bulat Lutfullin",
    author_email='blutfullin@hydrosphere.io',
    url="https://hydrosphere.io/",
    license="Apache 2.0",
    packages=pkgs,
    install_requires=["hydro-serving-grpc==2.1.0rc1", 'numpy'],
    include_package_data=True,
    setup_requires=['pytest-runner'],
    test_suite='tests',
    tests_require=[
        'pytest>=3.8.0', 'requests_mock>=1.5.0', 'mock>=2.0.0'
    ]
)
