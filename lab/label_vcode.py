# -*- coding: utf-8 -*-
"""用 ddddocr 给图片验证码样本打标签（多预处理变体 + 一致性）。

输出 lab/output/vcode/labels.json: {file: {text, variants:[...], n_agree}}
另生成 lab/output/vcode/contact_sheet.png 供人工抽查（24 张，标注文字画在图上）。

用法: python lab/label_vcode.py [--rebuild]
"""
import glob
import json
import os

import cv2
import ddddocr

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "samples", "vcode")
OUT = os.path.join(HERE, "output", "vcode")


def variants(img):
    out = [("raw", img)]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(("th", cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)))
    med = cv2.medianBlur(g, 3)
    up2 = cv2.resize(med, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    out.append(("med", cv2.cvtColor(up2, cv2.COLOR_GRAY2BGR)))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--incremental", action="store_true", help="只标注新增样本（保留已有 labels.json）")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ocr = ddddocr.DdddOcr(show_ad=False)
    files = sorted(glob.glob(os.path.join(SAMPLES, "*.jpg")))
    lab_path = os.path.join(OUT, "labels.json")
    rec = {}
    if args.incremental and os.path.exists(lab_path):
        rec = json.load(open(lab_path, encoding="utf-8"))
    todo = [f for f in files if os.path.basename(f) not in rec]
    print("images:", len(files), "| 已有:", len(rec), "| 待标注:", len(todo), flush=True)
    for k, f in enumerate(todo):
        img = cv2.imread(f)
        if img is None:
            continue
        labs = []
        for name, v in variants(img):
            try:
                labs.append(ocr.classification(cv2.imencode(".png", v)[1].tobytes()))
            except Exception:
                labs.append("")
        text = labs[0]
        if len(text) != 4:
            c4 = [l for l in labs if len(l) == 4]
            text = c4[0] if c4 else text
        agree = sum(1 for l in labs if l == text)
        rec[os.path.basename(f)] = {"text": text, "variants": labs, "n_agree": agree}
        if (k + 1) % 300 == 0:
            print("labeled", k + 1, flush=True)
    json.dump(rec, open(os.path.join(OUT, "labels.json"), "w", encoding="utf-8"), ensure_ascii=False)

    # 统计
    n4 = sum(1 for r in rec.values() if len(r["text"]) == 4)
    chars = {}
    for r in rec.values():
        if len(r["text"]) == 4:
            for c in r["text"]:
                chars[c] = chars.get(c, 0) + 1
    print("总样本:", len(rec), "| 4位标签:", n4, "(%.1f%%)" % (100.0 * n4 / max(1, len(rec))))
    print("字符集(%d):" % len(chars), "".join(sorted(chars)))
    print("平均一致变体数:", round(sum(r["n_agree"] for r in rec.values()) / max(1, len(rec)), 2))

    # 联系表：24 张随机样本 + 标签（人工核对用）
    import random
    random.seed(7)
    pick = random.sample([f for f in files if os.path.basename(f) in rec], min(24, len(files)))
    tiles = []
    for f in pick:
        img = cv2.imread(f)
        img = cv2.resize(img, (240, 102), interpolation=cv2.INTER_CUBIC)
        lab = rec[os.path.basename(f)]["text"]
        cv2.putText(img, lab, (6, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        tiles.append(img)
    rows = [cv2.hconcat(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(os.path.join(OUT, "contact_sheet.png"), cv2.vconcat(rows))
    print("联系表: lab/output/vcode/contact_sheet.png")


if __name__ == "__main__":
    main()
