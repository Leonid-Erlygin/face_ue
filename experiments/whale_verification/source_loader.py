"""Load specified pure functions/classes without importing training dependencies.

Only use this with your trusted local source tree. This executes the selected
Python definitions; it is not a sandbox for untrusted code.
"""
import ast
from pathlib import Path

def definitions(path, names, namespace):
    path = Path(path)
    tree = ast.parse(path.read_text(encoding='utf-8'))
    wanted = set(names)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    absent = wanted - {n.name for n in nodes}
    if absent:
        raise ValueError(f'{path}: missing definitions {sorted(absent)}')
    env = dict(namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env
