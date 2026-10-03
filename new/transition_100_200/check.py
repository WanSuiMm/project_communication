"""Focused metric fixtures plus immutable historical source/anchor checks."""
import argparse
from pathlib import Path
import subprocess
import sys

from run import HERE, ROOT, now, read, references, sources, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    out=Path(args.out).resolve()
    assert out.is_relative_to(ROOT/'analyses') and not out.exists()
    bindings=sources()
    refs=references()
    tests=subprocess.run([sys.executable,'-X','utf8','-B',str(HERE/'check_metrics.py')],
                         cwd=ROOT,text=True,encoding='utf-8',capture_output=True)
    assert tests.returncode==0,tests.stdout+tests.stderr
    import matplotlib
    assert sources()==bindings and references()==refs
    out.parent.mkdir(parents=True,exist_ok=True)
    write(out,{'status':'PASS','gpu_execution':False,'training':False,'finished_utc':now(),
               'source_sha256':bindings,'reference_bindings':refs,'metric_fixture_output':tests.stdout,
               'metric_fixture_stderr':tests.stderr,'matplotlib_version':matplotlib.__version__})
    print({'status':'PASS','source_bindings':len(bindings),'anchors':len(refs['anchors']),'output':str(out)})


if __name__=='__main__':
    main()
