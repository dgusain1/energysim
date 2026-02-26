# Copilot Instructions for energysim

## Virtual Environment

This project uses a Python virtual environment located at `.venv` in the workspace root.

**Always activate the venv before running any terminal command:**

```powershell
.\.venv\Scripts\activate
```

This applies to all operations: running tests, installing packages, executing scripts, etc.

## Common Commands

```powershell
# Install package in development mode
cd src; pip install -e .

# Run tests
python -m pytest test_energysim.py -v

# Install all optional dependencies
pip install -e ".[all]"
```

## Project Structure

- `src/energysim/` — Main package source code
- `src/setup.py` — Package setup / installation
- `test_energysim.py` — Test suite (workspace root)
- `examples/` — Example co-simulation scenarios
- `docs/` — Sphinx documentation

## Key Architecture

- **Coordinator**: `world` class in `src/energysim/__init__.py`
- **Base ABC**: `SimulatorAdapter` in `src/energysim/base.py`
- **Adapters**: `csAdapter`, `meAdapter`, `ppAdapter`, `pypsaAdapter`, `csv_adapter`, `signalAdapter`, `pyScriptAdapter`, `externalAdapter`, `matlabOctaveAdapter`, `pfAdapter`
- **Coupling modes**: `jacobi` (default), `gauss-seidel`, `iterative`
- **Core dependency**: `networkx` (used for dependency graph and topological ordering)
