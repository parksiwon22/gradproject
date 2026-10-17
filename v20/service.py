"""Reuse the V19 slow path, then refresh V20 audit even after collector failure."""
from v19.observer import run
from v20.core import audit,HOME

if __name__=='__main__':
    HOME.mkdir(parents=True,exist_ok=True)
    try:run()
    finally:audit()
