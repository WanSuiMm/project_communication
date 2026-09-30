"""Run the flat project as a package from a standalone repository checkout."""
import importlib.util
from pathlib import Path
import runpy
import sys


def load_package():
    root = Path(__file__).resolve().parent
    name = 'project_ReactionTransport'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, root / '__init__.py', submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return name


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {'check', 'smoke', 'qualify', 'plot'}:
        raise SystemExit('Usage: python run.py {check|smoke|qualify|plot} [arguments]')
    command = sys.argv[1]
    arguments = sys.argv[2:]
    module = {'check': 'test_transport', 'smoke': 'runner',
              'qualify': 'runner', 'plot': 'plot_results'}[command]
    if command in {'smoke', 'qualify'}:
        arguments = ['--mode', command, *arguments]
    sys.argv = [f'{module}.py', *arguments]
    runpy.run_module(f'{load_package()}.{module}', run_name='__main__')


if __name__ == '__main__':
    main()
