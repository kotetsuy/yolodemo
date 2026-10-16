# SPDX-License-Identifier: AGPL-3.0-only
from io import BytesIO
from zipfile import ZipFile
from app.source import source_archive


def test_source_offer_includes_notices_and_excludes_local_data_and_links(tmp_path):
    for name in ('app/server.py', 'LICENSE', 'NOTICE', 'licenses/Apache-2.0.txt',
                 'models/private.onnx', '.env', 'results/camera.png',
                 'app/__pycache__/private.py', '.venv/private.py'):
        path=tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    (tmp_path/'app/linked.py').symlink_to(tmp_path/'.env')
    (tmp_path/'scripts').symlink_to(tmp_path/'.venv', target_is_directory=True)
    with ZipFile(BytesIO(source_archive(tmp_path))) as archive:
        assert set(archive.namelist())=={'yolodemo/app/server.py','yolodemo/LICENSE',
                                      'yolodemo/NOTICE','yolodemo/licenses/Apache-2.0.txt'}


def test_license_and_source_http_routes_without_starting_inference():
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer
    from app.server import handler
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(object()))
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        base=f'http://127.0.0.1:{server.server_port}'
        with urllib.request.urlopen(base+'/license') as response:
            assert b'GNU AFFERO GENERAL PUBLIC LICENSE' in response.read()
        with urllib.request.urlopen(base+'/source.zip') as response:
            assert response.headers.get_content_type()=='application/zip'
            with ZipFile(BytesIO(response.read())) as archive:
                names=set(archive.namelist())
                assert {'yolodemo/LICENSE','yolodemo/NOTICE','yolodemo/app/server.py',
                        'yolodemo/licenses/Apache-2.0.txt',
                        'yolodemo/app/workers/npu_output_sync.cpp'}<=names
                assert not any('/models/' in name or '/.cache/' in name for name in names)
    finally:
        server.shutdown();server.server_close();thread.join(2)
