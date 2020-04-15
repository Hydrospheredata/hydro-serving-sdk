To populate hydrosdk automodule documentation run
```
sphinx-apidoc -f -e -o hydro_sdk_docs ../hydrosdk
```

To build documentation run
```
make clean && make markdown && make html
```

Built documentation is in `_build/markdown` and `_build/html` respectively


