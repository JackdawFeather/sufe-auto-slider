# -*- coding: utf-8 -*-
"""带 CORS/私网访问头的本地静态服务（供浏览器实测加载权重文件）。
用法: python cors_server.py [端口]   （在仓库根目录运行）
"""
import http.server, os, sys
from functools import partial

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Private-Network', 'true')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write('cors_server: ' + (fmt % args) + '\n')

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8777
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    handler = partial(Handler, directory=root)
    print('serving %s at http://127.0.0.1:%d' % (root, port))
    http.server.ThreadingHTTPServer(('127.0.0.1', port), handler).serve_forever()
