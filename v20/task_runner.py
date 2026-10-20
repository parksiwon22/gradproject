"""Visible, secret-redacted start/finish logging for Windows scheduled jobs."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import json
import importlib
import os
import re
import runpy
import sys
import traceback
import uuid

from v19.common import ROOT, now

HOME = ROOT / 'artifacts/v20/task_runs'


def write_once(path, record):
    """Atomic create-only JSON writer with Windows hardlink fallback."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, indent=2, default=str).encode('utf-8'))
            handle.flush(); os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except (OSError, NotImplementedError):
            temporary.rename(path)
    finally:
        if temporary.exists(): temporary.unlink()


def redact(text):
    text = str(text)
    for name, value in os.environ.items():
        if value and len(value) >= 4 and any(token in name.upper() for token in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'CREDENTIAL')):
            text = text.replace(value, '[REDACTED]')
    text = re.sub(r'(?i)(authKey|service[_-]?key|api[_-]?key|access[_-]?token|password)=([^&\s]+)', r'\1=[REDACTED]', text)
    return text[:1_000_000]


def run(module, args, home=HOME, launch=runpy.run_module):
    home.mkdir(parents=True, exist_ok=True)
    run_id = now().replace(':', '').replace('-', '') + '_' + uuid.uuid4().hex[:8]
    base = dict(run_id=run_id, module=module, arguments=list(args), started_at=now(), python=sys.executable,
                cwd=str(Path.cwd()), stage='python_started_module_not_yet_run')
    write_once(home / (run_id + '_started.json'), base)
    stdout, stderr = StringIO(), StringIO()
    exit_code, error_type, error_text = 0, None, None
    prior_argv = sys.argv[:]
    patched_writers = []
    try:
        # V19 imports a shared hardlink-only writer. In restricted Windows
        # folders os.link may be denied although same-directory rename works.
        for name in ('v12.records', 'v18.common', 'v19.common'):
            target = importlib.import_module(name)
            patched_writers.append((target, target.write_once))
            target.write_once = write_once
        sys.argv = [module, *args]
        with redirect_stdout(stdout), redirect_stderr(stderr):
            launch(module, run_name='__main__')
    except SystemExit as exc:
        code = exc.code
        exit_code = code if isinstance(code, int) else (0 if code is None else 1)
        if exit_code:
            error_type, error_text = 'SystemExit', str(code)
    except BaseException as exc:
        exit_code = 1
        error_type, error_text = type(exc).__name__, ''.join(traceback.format_exception(exc))
    finally:
        sys.argv = prior_argv
        for target, original in reversed(patched_writers):
            target.write_once = original
    output = redact(stdout.getvalue())
    errors = redact(stderr.getvalue())
    classification = ('success' if exit_code == 0 else 'module_import_failure' if error_type in ('ModuleNotFoundError', 'ImportError') or 'ModuleNotFoundError' in errors or 'ImportError' in errors else 'python_module_failure')
    result = {**base, 'completed_at': now(), 'stage': 'module_finished', 'outcome': classification,
              'exit_code': exit_code, 'exception_type': error_type, 'exception': redact(error_text or ''),
              'stdout': output, 'stderr': errors}
    write_once(home / (run_id + '_finished.json'), result)
    return exit_code


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) < 2 or argv[0] != '--module':
        print('Usage: python -m v20.task_runner --module PACKAGE [-- ARG ...]', file=sys.stderr)
        return 2
    module = argv[1]
    args = argv[2:]
    if args[:1] == ['--']:
        args = args[1:]
    return run(module, args)


if __name__ == '__main__':
    raise SystemExit(main())
