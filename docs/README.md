There are examples in `_build` folder, to build from scratch

Install sphinx `pip install -U sphinx`

Install reqs from `requirements.txt` file, but the list may be incomplete

To populate hydrosdk automodule documentation run
```
sphinx-apidoc -f -e -o hydro_sdk_docs ../hydrosdk
```

To build documentation run
```
make clean && make markdown && make html
```

Built documentation is in `_build/markdown` and `_build/html` respectively


