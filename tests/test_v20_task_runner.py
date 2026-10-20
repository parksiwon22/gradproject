import json
import os
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from v19.common import ROOT
from v20.task_runner import redact, run


class V20TaskRunnerTests(unittest.TestCase):
    def test_redacts_secret_env_and_url_values(self):
        with patch.dict(os.environ, {'KMA_APIHUB_AUTH_KEY': 'private-test-credential'}, clear=False):
            result = redact('authKey=private-test-credential&x=1 https://host/?serviceKey=anotherSecret')
        self.assertNotIn('private-test-credential', result)
        self.assertNotIn('anotherSecret', result)
        self.assertIn('[REDACTED]', result)

    def test_start_and_end_records_capture_module_failure_without_secrets(self):
        temp = ROOT / 'artifacts/v20' / ('test_task_runner_' + uuid.uuid4().hex)
        temp.mkdir(parents=True, exist_ok=True)
        try:
            def fail(*args, **kwargs):
                raise ModuleNotFoundError('missing optional package')
            with patch.dict(os.environ, {'KPX_SERVICE_KEY': 'private-test-credential'}, clear=False):
                code = run('private.test_module', ['--sample'], temp, launch=fail)
            self.assertEqual(code, 1)
            files = list(temp.glob('*.json'))
            self.assertEqual(len(files), 2)
            records = [json.loads(p.read_text(encoding='utf-8')) for p in files]
            start = next(r for r in records if r['stage'] == 'python_started_module_not_yet_run')
            finish = next(r for r in records if r['stage'] == 'module_finished')
            self.assertEqual(start['run_id'], finish['run_id'])
            self.assertEqual(finish['outcome'], 'module_import_failure')
            self.assertIn('ModuleNotFoundError', finish['exception'])
            self.assertNotIn('private-test-credential', finish['exception'])
        finally:
            for path in temp.iterdir():
                path.unlink()
            temp.rmdir()


if __name__ == '__main__':
    unittest.main()
