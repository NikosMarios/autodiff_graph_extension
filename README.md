# autodiff_graph_extension
Learning how neural networks, guided by automatic differentiation, recover the structure of symbolic mathematical expressions from data represented by  trees/graphs 

## Local setup

This repository uses the upstream project as a Git submodule:

```powershell
git submodule update --init --recursive
```

Create and activate a Python virtual environment from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The upstream reference project still lives in the submodule, but the
expression graph extension is copied into this repository under:

```powershell
cd math_expr
```

Run the expression graph demos and tests from there:

```powershell
..\.venv\Scripts\python.exe exp_expression_graph_demo.py
..\.venv\Scripts\python.exe exp_expression_graph_ad.py
..\.venv\Scripts\python.exe exp_expression_graph_visualize.py
..\.venv\Scripts\python.exe -m pytest test_expression_graph.py
```

The visualization script writes Mermaid diagrams and Graphviz PNG figures
under:

```powershell
math_expr\demo_outputs\figures
```

Graphviz PNG export requires the Python `graphviz` package and the
Graphviz system binary. If `dot.exe` is not on `PATH` after installation,
add the Graphviz `bin` directory, for example:

```powershell
$env:PATH='C:\Program Files\Graphviz\bin;' + $env:PATH
```

The original upstream scripts remain available in:

```powershell
cd external\stasinos_and_boura_repository\math_expr
..\..\..\.venv\Scripts\python.exe generator.py
```
