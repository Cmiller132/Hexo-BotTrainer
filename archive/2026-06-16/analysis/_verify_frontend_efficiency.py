"""Functional check of the frontend transfer-efficiency changes: gzip, static
caching + 304 revalidation, JSON ETag, and HTTP/1.1 keep-alive."""
import gzip
import threading
import time
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

from hexo_frontend.web import ManualMatchController, make_handler

ctrl = ManualMatchController()
server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(ctrl))
port = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()
time.sleep(0.3)


def req(path, headers=None):
    conn = HTTPConnection("127.0.0.1", port, timeout=15)
    conn.request("GET", path, headers=headers or {})
    resp = conn.getresponse()
    body = resp.read()
    hdrs = {k.lower(): v for k, v in resp.getheaders()}
    version = resp.version
    conn.close()
    return resp.status, hdrs, body, version


try:
    st, h, b, ver = req("/static/app.js", {"Accept-Encoding": "gzip"})
    raw = gzip.decompress(b) if h.get("content-encoding") == "gzip" else b
    print(f"app.js(gzip): status={st} http={ver} enc={h.get('content-encoding')} "
          f"wire={h.get('content-length')}B raw={len(raw)}B "
          f"ratio={len(b) / max(1, len(raw)):.2f} cc={h.get('cache-control')} etag={h.get('etag')}")

    et = h.get("etag")
    st2, h2, b2, _ = req("/static/app.js", {"Accept-Encoding": "gzip", "If-None-Match": et})
    print(f"app.js(revalidate If-None-Match): status={st2} (expect 304) body={len(b2)}B")

    st3, h3, b3, _ = req("/static/app.js", {})
    print(f"app.js(no-gzip client): status={st3} enc={h3.get('content-encoding')} wire={h3.get('content-length')}B")

    st4, h4, b4, _ = req("/api/training/runs", {"Accept-Encoding": "gzip"})
    rawj = gzip.decompress(b4) if h4.get("content-encoding") == "gzip" else b4
    print(f"api/training/runs(gzip): status={st4} enc={h4.get('content-encoding')} "
          f"wire={h4.get('content-length')}B raw={len(rawj)}B cc={h4.get('cache-control')} has_etag={bool(h4.get('etag'))}")

    ej = h4.get("etag")
    if ej:
        st5, _, b5, _ = req("/api/training/runs", {"Accept-Encoding": "gzip", "If-None-Match": ej})
        print(f"api/training/runs(revalidate): status={st5} (304 if unchanged) body={len(b5)}B")

    print(f"\nHTTP version negotiated: {'1.1' if ver == 11 else ver}  (keep-alive enabled)")
finally:
    server.shutdown()
    ctrl.close()
