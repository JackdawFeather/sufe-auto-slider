"""Fetch one captcha, decode images, report geometry."""
import base64, io, json, sys, urllib.request

BASE = "https://login.sufe.edu.cn/esc-sso"


def api(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method="POST" if data else "GET",
        headers={"Accept": "application/json",
                 **({"Content-Type": "application/json"} if data else {})},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "data/sample0"
    d = api("/api/v3/sliderCaptcha/init", {})["data"]
    print("Y =", d["Y"], " token =", d["token"])
    for name in ("sourceImage", "newImage"):
        raw = base64.b64decode(d[name])
        from PIL import Image
        im = Image.open(io.BytesIO(raw))
        print(f"{name}: size={im.size} mode={im.mode} b64len={len(d[name])} bytes={len(raw)}")
        im.save(f"{out}_{name}.png")
        if im.mode == "RGBA":
            a = im.getchannel("A")
            bb = a.getbbox()
            print(f"   alpha bbox={bb}  nonzero_alpha={sum(1 for p in a.getdata() if p > 0)}")
    with open(f"{out}_meta.json", "w", encoding="utf-8") as f:
        json.dump({"Y": d["Y"], "token": d["token"]}, f)


if __name__ == "__main__":
    main()
