import urllib.request, sys
try:
    req = urllib.request.Request('http://127.0.0.1:5000/api/status')
    with urllib.request.urlopen(req, timeout=10) as r:
        print('STATUS', r.status)
        print(r.read().decode())
except Exception as e:
    import traceback
    print('ERROR', e)
    traceback.print_exc()
    sys.exit(1)
