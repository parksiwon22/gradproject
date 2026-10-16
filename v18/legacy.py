"""Read-only issuance adapters: preserve prior reconciliation without new records."""
import argparse

def run(version):
    if version=='v12':
        from v12.forward import reconcile
        from v12.decisions import evaluate
        reconcile();evaluate()
    elif version=='v14':
        from v14.forward import reconcile
        reconcile()
    elif version=='v17':
        from v17.legacy import monitor
        monitor(allow_issue=False)
        from v17.forward import reconcile,status
        reconcile();status()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('version',choices=['v12','v14','v17']);run(p.parse_args().version)
