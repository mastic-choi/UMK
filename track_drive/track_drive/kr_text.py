#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# kr_text.py — cv2 디버그 창(cv2.imshow)에 한글 텍스트를 그리기 위한 유틸리티.
#
# ── 왜 cv2.putText로 바로 못 쓰나 ──
#   OpenCV 내장 폰트(Hershey 계열)는 라틴 문자만 지원한다. 한글 문자열을
#   cv2.putText에 그대로 넘기면 빈 사각형이거나 아예 안 그려진다. 그래서
#   PIL(Pillow)로 한글이 포함된 트루타입 폰트를 렌더링한 다음, 그 결과를 다시
#   cv2(BGR ndarray)로 합성해서 돌려준다.
#
# ── 실차 폰트 미설치 대비 ──
#   실제 온보드 컴퓨터(Jetson 등)에 한글 폰트가 깔려있는지 확인 전이라, 흔한 설치
#   경로를 순서대로 찾아보고 하나도 없으면(최초 1회만) 경고를 로깅한 뒤 호출부가
#   넘겨준 영문 fallback 문자열로 대신 그린다 — 창이 비는 것보다 로마자라도 보이는
#   게 디버깅에 낫다. 한글 폰트가 필요하면: sudo apt install fonts-nanum
#=============================================
import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False

_FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
    '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf',
    '/System/Library/Fonts/Supplemental/AppleGothic.ttf',  # macOS(개발/시뮬레이션 환경)용
    'C:/Windows/Fonts/malgun.ttf',                           # Windows(개발환경)용
]

_font_cache = {}
_font_path_cache = None
_font_path_searched = False
_warned_no_font = False


def _find_font_path():
    global _font_path_cache, _font_path_searched
    if _font_path_searched:
        return _font_path_cache
    _font_path_searched = True
    for p in _FONT_CANDIDATES:
        try:
            with open(p, 'rb'):
                _font_path_cache = p
                return p
        except OSError:
            continue
    return None


def _get_font(size):
    global _warned_no_font
    if not _PIL_AVAILABLE:
        if not _warned_no_font:
            print('[kr_text] Pillow(PIL) 미설치 — 한글 디버그 텍스트는 fallback(영문)만 표시됩니다.')
            _warned_no_font = True
        return None
    if size in _font_cache:
        return _font_cache[size]
    path = _find_font_path()
    if path is None:
        if not _warned_no_font:
            print('[kr_text] 한글 폰트를 찾지 못했습니다(NanumGothic/Noto CJK 등 미설치) — '
                  '한글 디버그 텍스트는 fallback(영문)만 표시됩니다. 설치 예: sudo apt install fonts-nanum')
            _warned_no_font = True
        _font_cache[size] = None
        return None
    font = ImageFont.truetype(path, size)
    _font_cache[size] = font
    return font


def put_text_kr(img_bgr, text, org, font_size=20, color_bgr=(255, 255, 255), fallback=None):
    """img_bgr(cv2 BGR ndarray) 위에 org=좌상단(x,y)부터 한 줄 한글 text를 그린다.
    한글 폰트를 못 찾으면 fallback(영문 문자열, 없으면 text 그대로)을 cv2.putText로
    대신 그린다. img_bgr을 제자리(in-place)에서 수정하고 그대로 반환한다."""
    return put_text_kr_multi(img_bgr, [(text, org, color_bgr, font_size, fallback)])


def put_text_kr_multi(img_bgr, lines):
    """여러 줄을 한 번의 PIL 왕복으로 그린다(줄마다 변환하면 낭비라서).
    lines: (text, org, color_bgr, font_size[, fallback]) 튜플의 리스트."""
    font = _get_font(20)  # 존재 여부만 확인 — 실제 크기별 폰트는 각 줄에서 개별 조회
    if not _PIL_AVAILABLE or font is None and _find_font_path() is None:
        for line in lines:
            text, org, color_bgr, font_size = line[0], line[1], line[2], line[3]
            fallback = line[4] if len(line) > 4 else None
            cv2.putText(img_bgr, fallback if fallback is not None else text,
                        org, cv2.FONT_HERSHEY_SIMPLEX, font_size / 30.0,
                        color_bgr, 1, cv2.LINE_AA)
        return img_bgr

    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    for line in lines:
        text, org, color_bgr, font_size = line[0], line[1], line[2], line[3]
        f = _get_font(font_size)
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        draw.text(org, text, font=f, fill=color_rgb)
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img_bgr[:, :, :] = result
    return img_bgr
