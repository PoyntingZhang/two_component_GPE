"""Colab HTTP/session boundary and end-to-end persistent run preparation."""
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from config.yaml_parser import new_config
from gui_periphery.editor.app import create_server

class ColabHttpTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.mount=Path(self.temp.name).resolve();(self.mount/'MyDrive').mkdir()
        self.output=self.mount/'MyDrive/simulations'
        for p in (patch.dict(os.environ,{'GPE_DRIVE_MOUNT':str(self.mount)}),patch('gpe.io.drive_storage.os.path.ismount',return_value=True)):
            p.start();self.addCleanup(p.stop)
        self.server=create_server(0,colab=True,output_parent=self.output)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self.close)
        self.url=f'http://127.0.0.1:{self.server.server_port}'

    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join()

    def request(self,path,data=None,token=True):
        headers={'Content-Type':'application/json','Host':'colab-proxy.example'}
        if token:headers['X-GPE-Session']=self.server.session_token
        return urlopen(Request(self.url+path,headers=headers,data=None if data is None else json.dumps(data).encode()),timeout=5)

    def test_proxy_requires_token_and_has_no_ssh_routes(self):
        with self.assertRaises(HTTPError) as error:self.request('/api/defaults',token=False)
        self.assertEqual(error.exception.code,403)
        with self.request('/api/defaults') as reply:
            self.assertIn(str(self.output),json.load(reply)['save_path'])
        with self.assertRaises(HTTPError) as error:self.request('/api/ssh/connect',{})
        self.assertEqual(error.exception.code,404)

    def test_prepare_through_http_captures_yaml_and_notes_on_drive(self):
        with self.request('/api/run/prepare',dict(config=new_config(),name='from_browser',notes='Two fields',output=str(self.output))) as reply:
            self.assertTrue(json.load(reply)['ok'])
        for _ in range(100):
            with self.request('/api/workflow') as reply:state=json.load(reply)
            if not state['operation']['busy']:break
            time.sleep(.01)
        self.assertFalse(state['operation']['error'])
        folder=self.output/'from_browser'
        self.assertTrue((folder/'config/run.yaml').exists())
        self.assertEqual((folder/'notes.txt').read_text(),'Two fields\n')
        with self.request('/api/run/download') as reply:self.assertTrue(reply.read().startswith(b'\x1f\x8b'))

    def test_yaml_save_rejects_ephemeral_target(self):
        with self.assertRaises(HTTPError) as error:self.request('/api/save',dict(config=new_config(),path='/content/lost.yaml'))
        self.assertEqual(error.exception.code,400)
