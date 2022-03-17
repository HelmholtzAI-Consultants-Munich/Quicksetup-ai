
## Steps to build your documentation

1. Build and install your package from source.
```bash
cd ./ML-Pipeline-Template/
pip install -e .
pip install sphinx sphinx_rtd_theme
```

2. Generate documentation from your docstrings.
```bash
cd docs/
sphinx-apidoc -f -o ./source ../src
```
3. Build the documentation
```bash
make clean && make html
```
4. You can now view your documentation under `docs/build/html/index.html`.
