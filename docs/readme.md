
## Steps to build Quicksetup-ai documentation

1. Build and install your package from source.
```bash
git clone https://github.com/HelmholtzAI-Consultants-Munich/Quicksetup-ai
cd Quicksetup-ai/
pip install -r docs/requirements.txt
```

2. Build the documentation
```bash
make clean && make html
```

3. You can now view your documentation under `docs/build/html/index.html`.
