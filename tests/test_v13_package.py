import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zipfile
from v13.package import scan,sanitized,pack_verify,verify_archive,copy,fs

class PackageTests(unittest.TestCase):
    def test_real_secret_and_private_key_rejected_without_echo(self):
        secret=b'PRIVATE_FAKE_KEY_FOR_TEST'
        with self.assertRaises(ValueError) as failure: scan(b'url?authKey='+secret,[secret])
        self.assertNotIn(secret.decode(),str(failure.exception))
        with self.assertRaises(ValueError): scan(b'-----BEGIN PRIVATE KEY-----\nexample\n',[])
        # Source code describing a PEM detector is not itself a private key.
        scan(b"    marker = b'-----BEGIN PRIVATE KEY-----'",[])
    def test_external_text_removes_personal_paths(self):
        self.assertNotIn('C:/Users/',sanitized('test C:/Users/Someone/private.txt end'))
    def test_zip_actual_extraction_inventory_and_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); stage=root/'stage'; stage.mkdir(); (stage/'README.md').write_text('shareable output')
            result=pack_verify(stage,root/'share.zip','EXTERNAL_SHARE',[])
            self.assertTrue(result['full_extraction_verified']); self.assertTrue(result['all_file_hashes_verified'])
            self.assertEqual(result['files'],2)
            extracted=Path(result['extracted_copy']); manifest=json.loads((extracted/'MANIFEST.json').read_text())
            self.assertEqual(manifest['files'][0]['sha256'],hashlib.sha256((extracted/'README.md').read_bytes()).hexdigest())
    def test_sqlite_copy_includes_wal_and_preserves_source_rows(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'live.sqlite'; connection=sqlite3.connect(source)
            connection.execute('PRAGMA journal_mode=WAL'); connection.execute('CREATE TABLE events(x)'); connection.execute('INSERT INTO events VALUES (42)'); connection.commit()
            destination=root/'snapshot.sqlite'; copy(source,destination,[])
            snapshot=sqlite3.connect(destination); self.assertEqual(snapshot.execute('SELECT x FROM events').fetchone()[0],42)
            self.assertEqual(connection.execute('SELECT x FROM events').fetchone()[0],42)
            snapshot.close(); connection.close()
    def test_windows_long_archive_paths_extract_and_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); archive=root/'long.zip'; name='RESTORE/'+('directory_'*12)+'/'+('forecast_'*15)+'.json'; body=b'{"preserved":true}'
            manifest=dict(kind='PRIVATE_RECOVERY_DO_NOT_SHARE',files=[dict(path=name,bytes=len(body),sha256=hashlib.sha256(body).hexdigest())])
            with zipfile.ZipFile(archive,'x') as z:
                z.writestr(name,body); z.writestr('MANIFEST.json',json.dumps(manifest))
            result=verify_archive(archive,manifest,[])
            self.assertEqual(fs(Path(result['extracted_copy'])/name).read_bytes(),body)

if __name__=='__main__': unittest.main()
