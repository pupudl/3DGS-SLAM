#!/usr/bin/env python3
"""Export saved sequence diagnostics as numbered PNGs and PDF walkthroughs.

No estimation, feature matching, CUDA rendering, or HTML generation occurs.
Local-motion labels, global-object identities, and visible pixel ownership are
kept distinct. Manual focus rectangles only select diagnostic views.
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

EXPERIMENT = Path(__file__).resolve().parents[2]
ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(EXPERIMENT))
from ownership.object_masks import object_color

DIAGNOSTICS = EXPERIMENT / 'diagnostics' / 'missed_dynamic_objects_20261114'
RUNS = [
    ('kitti360_2439_2466_unified', '01_cyclist', '第一组 · 骑车人', [2454, 2460, 2463]),
    ('kitti360_drive0002_4914_4934', '02_red_car', '第二组 · 红色汽车', [4914, 4920, 4923]),
    ('kitti360_drive0004_0776_0796', '03_gray_car', '第三组 · 灰色汽车', [779, 782, 785]),
]
STEPS = [
    ('01_all_matches.png', '01 全图匹配'),
    ('02_target_matches.png', '02 目标附近的匹配'),
    ('03_motion_support.png', '03 运动支持与误差'),
    ('04_feedback.png', '04 反馈补匹配'),
    ('05_objects_and_ownership.png', '05 object 关联与最终归属'),
]
PALETTE = np.rint(np.array([
    [.18, .68, 1.], [1., .52, .16], [.30, .88, .42], [.92, .37, .82],
    [1., .85, .24], [.28, .86, .87], [.75, .56, 1.], [1., .47, .52],
]) * 255).astype(np.uint8)
W, PAD, GAP = 1536, 32, 24
CELL = (W - 2 * PAD - GAP) // 2
INK, MUTED, ORANGE = (25, 35, 45), (86, 96, 105), (226, 125, 22)
FONT_PATH = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if not FONT_PATH.exists():
    FONT_PATH = Path('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
FONTS = {n: ImageFont.truetype(str(FONT_PATH), n) for n in [18, 20, 22, 24, 26, 30, 36]}


def text(image, xy, value, size=24, fill=INK):
    ImageDraw.Draw(image).text(xy, str(value), font=FONTS[size], fill=fill)


def sheet(title, subtitle, height):
    image = Image.new('RGB', (W, height), 'white')
    text(image, (PAD, 17), title, 36)
    text(image, (PAD, 69), subtitle, 22, MUTED)
    return image


def rgb(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def inside(points, box):
    if box is None:
        return np.zeros(len(points), bool)
    x0, y0, x1, y1 = box
    return ((points[:, 0] >= x0) & (points[:, 0] < x1) &
            (points[:, 1] >= y0) & (points[:, 1] < y1))


def motion_color(label):
    return tuple(map(int, PALETTE[int(label) % len(PALETTE)])) if label >= 0 else (175, 175, 175)


def oid_color(oid):
    if oid == -2:
        return (20, 20, 20)
    if oid == -1:
        return (155, 155, 155)
    return tuple(reversed(object_color(int(oid))))


def pair_color(index):
    return tuple(round(c * 255) for c in colorsys.hsv_to_rgb((index * .61803398875) % 1, .72, 1))


def count_labels(values):
    return {str(int(k)): int(v) for k, v in zip(*np.unique(values, return_counts=True))}


def marked_box(image, box):
    result = image.copy()
    if box is not None:
        cv2.rectangle(result, tuple(box[:2]), tuple(box[2:]), ORANGE, 2, cv2.LINE_AA)
    return result


def put(image, array, x, y, width=CELL, height=None, nearest=False):
    original = Image.fromarray(np.asarray(array, np.uint8))
    if height is None:
        height = round(original.height * width / original.width)
    scale = min(width / original.width, height / original.height)
    resized = original.resize((max(1, round(original.width * scale)), max(1, round(original.height * scale))),
                              Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS)
    offset = (x + (width - resized.width) // 2, y + (height - resized.height) // 2)
    image.paste(resized, offset)
    return scale, offset


def caption_pair(page, left, right, y, labels, height=None, nearest=False):
    text(page, (PAD, y), labels[0], 24)
    text(page, (PAD + CELL + GAP, y), labels[1], 24)
    y += 39
    put(page, left, PAD, y, height=height, nearest=nearest)
    put(page, right, PAD + CELL + GAP, y, height=height, nearest=nearest)
    return y + (height if height is not None else round(left.shape[0] * CELL / left.shape[1]))


def crop_box(shape, box, points=None, padding=24):
    h, w = shape[:2]
    if box is None:
        return (0, 0, w, h)
    low = np.array(box[:2], float)
    high = np.array(box[2:], float)
    if points is not None and len(points):
        finite = points[np.isfinite(points).all(1)]
        if len(finite):
            low = np.minimum(low, finite.min(0))
            high = np.maximum(high, finite.max(0))
    center = .5 * (low + high)
    size = np.maximum(high - low + 2 * padding, [95, 80])
    # Preserve at least 1.7:1 horizontal context without removing endpoints.
    size[0] = max(size[0], size[1] * 1.7)
    low = np.maximum(0, np.floor(center - size / 2)).astype(int)
    high = np.minimum([w, h], np.ceil(center + size / 2)).astype(int)
    return (*map(int, low), *map(int, high))


def crop(array, box):
    return array[box[1]:box[3], box[0]:box[2]].copy()


def motion_crop(image, box, points, labels):
    """Draw fixed-size markers after magnification so they do not cover the object."""
    local = cv2.convertScaleAbs(crop(image, box), alpha=.8)
    scale = min(CELL / local.shape[1], 285 / local.shape[0])
    size = (max(1, round(local.shape[1] * scale)), max(1, round(local.shape[0] * scale)))
    result = np.full((285, CELL, 3), 255, np.uint8)
    offset = np.array([(CELL - size[0]) // 2, (285 - size[1]) // 2])
    result[offset[1]:offset[1] + size[1], offset[0]:offset[0] + size[0]] = cv2.resize(local, size)
    selected = inside(points, box)
    for point, label in zip(points[selected], labels[selected]):
        xy = tuple(np.rint((point - np.asarray(box[:2])) * scale + offset).astype(int))
        if label < 0:
            cv2.circle(result, xy, 3, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.circle(result, xy, 1, (145, 145, 145), -1, cv2.LINE_AA)
        else:
            cv2.circle(result, xy, 4, (12, 12, 12), -1, cv2.LINE_AA)
            cv2.circle(result, xy, 3, motion_color(int(label)), -1, cv2.LINE_AA)
    return result


def index_of_matches(reference, selected):
    keys = {tuple(row): i for i, row in enumerate(reference)}
    return np.asarray([keys.get(tuple(row), -1) for row in selected], np.int64)


def save(page, path):
    page.save(path, optimize=True)


def table(page, y, headers, rows, widths, row_height=38, colors=None, size=22):
    draw = ImageDraw.Draw(page)
    x = PAD
    for heading, width in zip(headers, widths):
        text(page, (x, y), heading, size, MUTED)
        x += width
    y += row_height
    draw.line((PAD, y - 4, W - PAD, y - 4), fill=(215, 220, 224), width=1)
    for i, row in enumerate(rows):
        x = PAD
        for j, (value, width) in enumerate(zip(row, widths)):
            if j == 0 and colors:
                draw.ellipse((x, y + 8, x + 15, y + 23), fill=colors[i])
                text(page, (x + 25, y), value, size)
            else:
                text(page, (x, y), value, size)
            x += width
        y += row_height
    return y


def all_matches_page(images, matches, metric_rows, boxes, title):
    h, w = images[0].shape[:2]
    gutter = 80
    canvas = np.full((2 * h + gutter, w, 3), 248, np.uint8)
    canvas[:h] = marked_box(images[0], boxes[0])
    canvas[h + gutter:] = marked_box(images[1], boxes[1])
    selection = np.unique(np.linspace(0, len(matches) - 1, min(160, len(matches))).astype(int))
    tint = canvas.copy()
    for i in selection:
        a, b = np.rint(matches[i, :2]).astype(int), np.rint(matches[i, 2:]).astype(int) + [0, h + gutter]
        cv2.line(tint, tuple(a), tuple(b), pair_color(i), 1, cv2.LINE_AA)
    canvas = cv2.addWeighted(canvas, .6, tint, .4, 0)
    metric = metric_rows >= 0
    for i, row in enumerate(matches):
        for shift, point in [(0, row[:2]), (h + gutter, row[2:])]:
            xy = tuple(np.rint(point).astype(int) + [0, shift])
            cv2.circle(canvas, xy, 2, (80, 230, 150) if metric[i] else (230, 230, 230),
                       -1 if metric[i] else 1, cv2.LINE_AA)
    page = sheet(title + '  /  01 全图匹配',
                 f'初始 2D 匹配 {len(matches)} 对；可进入三维估计 {int(metric.sum())} 对；连线均匀抽样 {len(selection)} 对。', 1170)
    text(page, (PAD, 112), '上：源帧；下：目标帧。绿实心＝有效三维匹配；白空心＝仅二维。线条颜色只区分对应关系。', 22)
    put(page, canvas, PAD, 160, width=W - 2 * PAD)
    text(page, (PAD, 1098), '橙框仅供人工定位，包含周围背景，不参与原算法；三维有效不等于匹配正确。', 22, MUTED)
    return page


def target_matches_page(images, raw, metric_rows, boxes, title):
    selected = np.flatnonzero(inside(raw[:, :2], boxes[0]))
    no_focus = boxes[0] is None
    if no_focus:
        selected = np.unique(np.linspace(0, len(raw) - 1, min(30, len(raw))).astype(int))
    matched = raw[selected]
    page = sheet(title + '  /  02 目标附近的匹配',
                 ('本帧目标已出画：下面显示 30 对均匀抽样匹配。' if no_focus else
                  f'源帧人工框选出 {len(selected)} 对二维匹配，其中 {int((metric_rows[selected] >= 0).sum())} 对有有效三维。'), 1130)
    y = caption_pair(page, marked_box(images[0], boxes[0]), marked_box(images[1], boxes[1]), 112,
                     ('源帧：观察目标的位置', '目标帧：观察目标移动到哪里'))
    crops = [crop_box(im.shape, box, matched[:, 2 * j:2 * j + 2])
             for j, (im, box) in enumerate(zip(images, boxes))]
    y += 22
    text(page, (PAD, y), '局部放大：实心点＝有效三维；空心点＝仅二维；同号同色是一对匹配，不表示正确。', 22)
    y += 40
    draw = ImageDraw.Draw(page)
    positions = []
    image_y = y + 36
    for j, (im, box) in enumerate(zip(images, boxes)):
        x = PAD + j * (CELL + GAP)
        text(page, (x, y), '源帧局部' if j == 0 else '目标端匹配落点（保留框外落点）', 24)
        arr = marked_box(im, box)
        cb = crops[j]
        scale, offset = put(page, crop(arr, cb), x, image_y, height=410)
        positions.append((scale, offset, cb))
    # All selected pairs are drawn. Labels are bounded to keep dense crops readable.
    label_indices = set(np.linspace(0, max(len(selected) - 1, 0), min(20, len(selected))).astype(int))
    for local, i in enumerate(selected):
        coords = []
        for j in [0, 1]:
            scale, offset, cb = positions[j]
            pt = (raw[i, 2 * j:2 * j + 2] - np.array(cb[:2])) * scale + offset
            coords.append(tuple(map(float, pt)))
        c = pair_color(int(i))
        draw.line(coords, fill=tuple(round(.55 * a + .45 * 255) for a in c), width=1)
        for xy in coords:
            radius = 5
            draw.ellipse((xy[0] - radius, xy[1] - radius, xy[0] + radius, xy[1] + radius),
                         fill=c if metric_rows[i] >= 0 else None, outline=(10, 10, 10), width=2)
            if metric_rows[i] < 0:
                draw.ellipse((xy[0] - 4, xy[1] - 4, xy[0] + 4, xy[1] + 4), outline=c, width=2)
            if local in label_indices:
                draw.text((xy[0] + 6, xy[1] - 20), str(local + 1), font=FONTS[18], fill=c,
                          stroke_width=2, stroke_fill=(15, 15, 15))
    y = image_y + 435
    text(page, (PAD, y), '阅读顺序：先找到源帧车身/骑车人的点，再沿线看目标端是否仍落在同一物体上。', 24)
    text(page, (PAD, y + 42), '目标框外的点不能直接判错：物体边缘、框的精度、遮挡和运动都会影响框内外关系。', 22, MUTED)
    text(page, (PAD, y + 84), '局部点编号对应 matches.tsv 的 focus_index；表格中另有原始编号、像素坐标、深度和代价。', 22, MUTED)
    return page, selected, crops


def motion_support_page(images, initial, labels, costs, H, K, xyz0, boxes, title):
    focus = inside(initial[:, :2], boxes[0])
    if boxes[0] is None:
        focus[:] = True
    counts = np.bincount(labels[focus & (labels >= 0)], minlength=len(H))
    chosen = (int(counts.argmax()) if counts.max(initial=0) > 0 else
              int(np.median(costs[focus], axis=0).argmin()) if focus.any() else 0)
    page = sheet(title + '  /  03 运动支持与误差',
                 f'保存的两帧初始化：{len(initial)} 对三维匹配；关注区域 {int(focus.sum())} 对；灰点 {int((focus & (labels < 0)).sum())} 对。', 1460)
    cb = [crop_box(im.shape, box, initial[focus, 2 * j:2 * j + 2])
          for j, (im, box) in enumerate(zip(images, boxes))]
    dots = [motion_crop(im, cb[j], initial[:, 2 * j:2 * j + 2], labels) for j, im in enumerate(images)]
    y = caption_pair(page, dots[0], dots[1], 112,
                     ('源帧运动支持点', '同一批支持点在目标帧的位置'), height=285)
    text(page, (PAD, y + 14), '彩色点＝通过该局部运动的竞争/三维核心检查；灰圈＝未分配。L 编号仅在本帧对内有效。', 22)
    y += 62
    rows = []
    for k in range(len(H)):
        median = f'{np.median(costs[focus, k]):.3f}' if focus.any() else '—'
        rows.append((f'L{k}' + (' 参考' if k == 0 else ''), int((labels == k).sum()), int(counts[k]), median))
    y = table(page, y, ['局部运动', '全图核心支持', '关注区域核心支持', '区域中位代价（<1 才可能通过）'],
              rows, [230, 260, 340, 620], colors=[motion_color(k) for k in range(len(H))])
    text(page, (PAD, y + 7), '代价 <1 还需通过竞争 margin 和未截断三维一致性；这不是物体类别或真值标注。', 22, MUTED)
    y += 55
    moved = xyz0 @ H[chosen, :3, :3].T + H[chosen, :3, 3]
    project = moved @ K.T
    prediction = project[:, :2] / np.maximum(project[:, 2:], 1e-8)
    in_frame = ((moved[:, 2] > 0) & np.isfinite(prediction).all(1) &
                (prediction[:, 0] >= 0) & (prediction[:, 0] < images[1].shape[1]) &
                (prediction[:, 1] >= 0) & (prediction[:, 1] < images[1].shape[0]))
    forward = cv2.convertScaleAbs(images[1], alpha=.76)
    for i in np.flatnonzero(focus):
        obs = tuple(np.rint(initial[i, 2:]).astype(int))
        if in_frame[i]:
            pred = tuple(np.rint(prediction[i]).astype(int))
            cv2.line(forward, obs, pred, (255, 178, 35), 1, cv2.LINE_AA)
            cv2.drawMarker(forward, pred, (255, 178, 35), cv2.MARKER_CROSS, 9, 2)
        cv2.circle(forward, obs, 3, (85, 220, 255), -1, cv2.LINE_AA)
    text(page, (PAD, y), f'区域支持最多/代价最小的代表运动 L{chosen}：蓝点＝实测，橙十字＝预测，线段＝像素误差', 24)
    valid = focus & in_frame
    err = np.linalg.norm(prediction[valid] - initial[valid, 2:], axis=1)
    error_text = f'{np.median(err):.2f} px' if len(err) else '无有效投影'
    text(page, (PAD, y + 38), f'关注区域前向像素误差中位数 {error_text}；{int((focus & ~in_frame).sum())} 个预测落在图外或相机后方。', 22, MUTED)
    put(page, forward, PAD, y + 85, width=W - 2 * PAD)
    return page


def feedback_page(images, pair_dir, initial, final, boxes, title):
    page = sheet(title + '  /  04 反馈补匹配',
                 '按保存的反馈轮次查看：黄色为实际选择的目标端问题区域；红点为该轮新增三维匹配。', 1220)
    files = sorted(pair_dir.glob('feedback_evidence_cycle_*.npz'))
    y = 118
    summary = []
    for file in files:
        cycle = int(file.stem.split('_')[-1])
        with np.load(file) as z:
            mask, novel, raw, active = z['problem_mask'], z['novel_matches'], z['raw_matches'], z['active_matches']
        im = images[1].copy()
        im[mask] = np.rint(.57 * im[mask] + .43 * np.array([255, 195, 40])).astype(np.uint8)
        for xy in initial[:, 2:]:
            cv2.circle(im, tuple(np.rint(xy).astype(int)), 2, (145, 145, 145), -1, cv2.LINE_AA)
        for xy in novel[:, 2:]:
            cv2.circle(im, tuple(np.rint(xy).astype(int)), 4, (250, 55, 55), -1, cv2.LINE_AA)
        im = marked_box(im, boxes[1])
        target_focus = inside(novel[:, 2:], boxes[1])
        local_box = crop_box(im.shape, boxes[1])
        text(page, (PAD, y), f'第 {cycle + 1} 轮：原始匹配 {len(raw)}，新增三维 {len(novel)}，进入本轮评估 {len(active)}', 24)
        y += 38
        put(page, im, PAD, y, height=210)
        put(page, crop(im, local_box), PAD + CELL + GAP, y, height=210)
        text(page, (PAD + CELL + GAP, y + 215), f'目标框内新增 {int(target_focus.sum())} 对（不等于最终保留）', 22, MUTED)
        summary.append(dict(cycle=cycle, raw=len(raw), novel=len(novel), active=len(active),
                            target_box_novel=int(target_focus.sum())))
        y += 275
    initial_keys = {tuple(row) for row in initial}
    retained = np.array([tuple(row) not in initial_keys for row in final])
    text(page, (PAD, y + 4), f'帧对结束：保留三维匹配 {len(final)} 对，其中相对初始化新增 {int(retained.sum())} 对。', 24)
    text(page, (PAD, y + 47), '本实验冻结局部归属并关闭 pair 内 birth：补到了点，并不等于新运动已经出生。', 22, MUTED)
    return page, summary


def pixel_map(shape, uv, values, fill=-2):
    result = np.full(shape[:2], fill, dtype=np.int32)
    xy = np.rint(uv).astype(int)
    keep = ((xy[:, 0] >= 0) & (xy[:, 0] < shape[1]) &
            (xy[:, 1] >= 0) & (xy[:, 1] < shape[0]))
    result[xy[keep, 1], xy[keep, 0]] = np.asarray(values)[keep]
    return result


def label_overlay(image, labels, alpha=.6):
    result = image.copy()
    for oid in np.unique(labels):
        selected = labels == oid
        if oid == -2:
            continue
        result[selected] = np.rint((1 - alpha) * image[selected] + alpha * np.asarray(oid_color(oid))).astype(np.uint8)
    return result


def flag_overlay(image, mask, col):
    result = cv2.convertScaleAbs(image, alpha=.65)
    result[mask] = np.rint(.2 * image[mask] + .8 * np.array(col)).astype(np.uint8)
    return result


def objects_page(image, initial_labels, tr_labels, mapping, state, temporal, owner, box, title, policy_history):
    page = sheet(title + '  /  05 object 关联与最终归属',
                 'L＝本帧对的局部运动；O＝跨帧 object。右侧最终归属来自已保存的联合可见性结果。', 1510)
    saved = json.loads(str(state['report_json']))
    connectivity = saved.get('render_optimization', {}).get('spatial_constraints', {}).get('connectivity', {})
    cores = connectivity.get('graph', {}).get('objects', {})
    disabled = set(connectivity.get('no_core_fallback', {}).get('disabled_object_ids', []))
    removed = set(saved.get('empty_object_pruning', {}).get('removed_object_ids', []))
    model_info = {row['object_id']: row for row in temporal.get('object_window', {}).get('models', [])}
    rows = []
    for k, oid in enumerate(mapping):
        oid = int(oid)
        current = model_info.get(oid)
        model_text = ('有效' if current['valid'] else '无效') if current else '见原记录'
        historical = policy_history.get(str(oid), {})
        core = historical.get('core', cores.get(str(oid), {}))
        core_text = '背景免检' if oid == 0 else str(core.get('core_count', '—'))
        rejected = oid in disabled or oid in removed or historical.get('disabled', False)
        action = '曾缺核禁用/移除' if rejected else '未见缺核淘汰'
        rows.append((f'L{k} → O{oid}', int((initial_labels == k).sum()), int((tr_labels == k).sum()), model_text, core_text, action))
    y = table(page, 116, ['帧对结束时映射', '初始支持', '结束支持', '当时 object 运动', '后续核心', '窗口历史中的处理'],
              rows, [265, 160, 160, 250, 165, 400], row_height=35,
              colors=[motion_color(k) for k in range(len(mapping))], size=22)
    text(page, (PAD, y + 2), '核心与淘汰状态合并后续窗口历史；运动有效性取本帧对当时记录。', 20, MUTED)
    y += 39
    uv = state['uv']
    before = state['evidence_object_ids'][state['q_object_initial'].argmax(1)]
    before_map = pixel_map(image.shape, uv, before)
    before_overlay = marked_box(label_overlay(image, before_map), box)
    after_overlay = marked_box(label_overlay(image, owner), box)
    y = caption_pair(page, before_overlay, after_overlay, y,
                     ('object 初始化 argmax（无拒绝阈值）', '最终可见归属：灰＝unknown，彩色＝O ID'))
    cb = crop_box(image.shape, box)
    y = caption_pair(page, crop(before_overlay, cb), crop(after_overlay, cb), y + 8,
                     ('初始化局部放大', '最终局部放大'), height=220, nearest=True)
    unique = sorted(set(map(int, np.unique(before))) | set(map(int, np.unique(owner))))
    draw = ImageDraw.Draw(page)
    x, yy = PAD, y + 10
    for oid in unique:
        if oid < -1:
            continue
        draw.rectangle((x, yy + 4, x + 18, yy + 23), fill=oid_color(oid))
        text(page, (x + 27, yy - 2), f'O{oid}' if oid >= 0 else 'unknown -1', 20)
        x += 110 if oid >= 0 else 200
    y += 60
    height = pixel_map(image.shape, uv, state['height_forced_background'].astype(np.int32), fill=0).astype(bool)
    forced = pixel_map(image.shape, uv, state['connectivity_forced_background'].astype(np.int32), fill=0).astype(bool)
    hf = flag_overlay(image, height, (255, 157, 28))
    cf = flag_overlay(image, forced, (240, 52, 78))
    y = caption_pair(page, hf, cf, y,
                     (f'高度强制背景：{int(state["height_forced_background"].sum()):,} 个高斯',
                      f'连通性强制背景：{int(state["connectivity_forced_background"].sum()):,} 个高斯'))
    text(page, (PAD, y + 14), '下排仅显示被强制改成背景的位置，按源像素投影；上排右图读取已有的可见性遮挡结果。', 20, MUTED)
    text(page, (PAD, y + 46), '黑色/未着色区域可能是天空或没有可见高斯；O0 是背景，unknown(-1) 是未确认归属。', 20, MUTED)
    # Height depends on model count; trim rather than leaving a blank tail or clipping footnotes.
    if y + 88 > page.height:
        raise ValueError(f'Object page content exceeds canvas: {title}, y={y}')
    return page.crop((0, 0, W, y + 88))


def export_pair(run, output, boxes, seq_title, policy_history):
    temporal = json.loads((run / 'temporal_state.json').read_text())
    first, second = int(temporal['first']), int(temporal['second'])
    cfg = yaml.safe_load((run / 'config.yaml').read_text())
    data_root = Path(cfg['dataset']['data_root'])
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    seq = data_root / 'data_2d_raw' / cfg['dataset']['sequence']
    images = [rgb(seq / 'image_00/data_rect' / f'{f:010d}.png') for f in [first, second]]
    focus_boxes = [boxes.get(str(first)), boxes.get(str(second))]
    output.mkdir(parents=True, exist_ok=True)
    title = f'{seq_title}   {first} → {second}'
    with np.load(run / 'pairwise_state.npz') as z:
        raw, initial = z['initial_matches_2d'], z['anchors_initial']
        labels, costs, H = z['initial_motion_labels'], z['initial_motion_costs'], z['H_sparse']
    assert labels.shape == (len(initial),) and costs.shape == (len(initial), len(H))
    metric_rows = index_of_matches(initial, raw)
    assert np.count_nonzero(metric_rows >= 0) == len(initial), 'Initial 3D matches must be a subset of 2D matches'
    with np.load(run / 'pair_tracks.npz') as z:
        final, tr_labels, mapping, tracks = z['matches'], z['labels'], z['local_to_object'], z['track_ids']
    assert len(mapping) == len(H)
    with np.load(run / 'object_state.npz') as z:
        names = ['uv', 'object_labels', 'q_object_initial', 'evidence_object_ids', 'height_forced_background',
                 'connectivity_forced_background', 'K', 'report_json']
        state = {name: z[name] for name in names}
    with np.load(run.parents[2] / 'objects/visibility' / f'frame_{first:010d}.npz') as z:
        owner = z['owner']
    assert owner.shape == images[0].shape[:2]
    depth0 = np.load(seq / cfg['depth'].get('subdir', 'depth_sceneflow') / f'{first:010d}.npy')
    depth1 = np.load(seq / cfg['depth'].get('subdir', 'depth_sceneflow') / f'{second:010d}.npy')
    xy0, xy1 = np.rint(initial[:, :2]).astype(int), np.rint(initial[:, 2:]).astype(int)
    d0, d1 = depth0[xy0[:, 1], xy0[:, 0]], depth1[xy1[:, 1], xy1[:, 0]]
    K = state['K']
    xyz0 = np.column_stack(((initial[:, 0] - K[0, 2]) * d0 / K[0, 0],
                            (initial[:, 1] - K[1, 2]) * d0 / K[1, 1], d0))
    save(all_matches_page(images, raw, metric_rows, focus_boxes, title), output / STEPS[0][0])
    page, selected, _ = target_matches_page(images, raw, metric_rows, focus_boxes, title)
    save(page, output / STEPS[1][0])
    save(motion_support_page(images, initial, labels, costs, H, K, xyz0, focus_boxes, title), output / STEPS[2][0])
    page, feedback = feedback_page(images, run, initial, final, focus_boxes, title)
    save(page, output / STEPS[3][0])
    page = objects_page(images[0], labels, tr_labels, mapping, state, temporal, owner, focus_boxes[0], title, policy_history)
    save(page, output / STEPS[4][0])
    # The raw-match index is stable across figures and the exported numeric table.
    with (output / 'matches.tsv').open('w', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['match_index', 'focus_index', 'source_u', 'source_v', 'target_u', 'target_v', 'in_source_focus',
                         'in_target_focus', 'has_valid_3d', 'source_depth_m', 'target_depth_m', 'initial_local_label',
                         'background_cost', 'best_cost', 'nearest_cost_model'])
        src_focus, dst_focus = inside(raw[:, :2], focus_boxes[0]), inside(raw[:, 2:], focus_boxes[1])
        focus_number = {int(index): j + 1 for j, index in enumerate(selected)}
        for i, row in enumerate(raw):
            j = metric_rows[i]
            extra = [float(d0[j]), float(d1[j]), int(labels[j]), float(costs[j, 0]),
                     float(costs[j].min()), int(costs[j].argmin())] if j >= 0 else [''] * 6
            writer.writerow([i, focus_number.get(i, ''), *map(float, row), int(src_focus[i]), int(dst_focus[i]), int(j >= 0), *extra])
    meta = dict(first=first, second=second, source_result=str(run), diagnostic_boxes=focus_boxes,
                initial_2d_matches=len(raw), initial_3d_matches=len(initial), final_3d_matches=len(final),
                source_focus_2d_matches=int(inside(raw[:, :2], focus_boxes[0]).sum()),
                source_focus_3d_matches=int(inside(initial[:, :2], focus_boxes[0]).sum()),
                initial_supports=count_labels(labels), mapping=mapping.tolist(), feedback=feedback,
                figure_steps=[name for name, _ in STEPS],
                interpretation='Diagnostic rectangles include background and do not judge matching correctness; '
                               'initialization includes temporal warm fitting; global IDs are not local model IDs.')
    (output / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    return meta


def export_pdf(paths, destination):
    pages = [Image.open(path).convert('RGB') for path in paths]
    pages[0].save(destination, 'PDF', save_all=True, append_images=pages[1:], resolution=120, quality=86)
    for page in pages:
        page.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DIAGNOSTICS / 'step_by_step')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit = json.loads((DIAGNOSTICS / 'saved_output_audit.json').read_text())
    manifest = []
    for name, folder, title, key_frames in RUNS:
        destination = args.output / folder
        sequence = []
        policy_history = {}
        for window in audit[name]['windows']:
            disabled = set(window.get('fallback', {}).get('disabled_object_ids', []))
            for oid, core in (window.get('cores') or {}).items():
                if policy_history.get(oid, {}).get('disabled'):
                    continue
                policy_history[oid] = dict(core=core, disabled=int(oid) in disabled, frames=window['frames'])
        for pair in sorted((EXPERIMENT / 'outputs' / name / 'pairs').glob('*/multi_frame')):
            frame0, frame1 = map(int, pair.parent.name.split('_')[-2:])
            sub = destination / f'{frame0:04d}_{frame1:04d}'
            sequence.append(export_pair(pair, sub, audit[name]['boxes'], title, policy_history))
            print(f'{folder}: {frame0} -> {frame1}: 5 PNGs + matches.tsv', flush=True)
        paths = [destination / f'{row["first"]:04d}_{row["second"]:04d}' / step for row in sequence for step, _ in STEPS]
        export_pdf(paths, destination / 'all_pairs.pdf')
        key_paths = [destination / f'{row["first"]:04d}_{row["second"]:04d}' / step for row in sequence
                     if row['first'] in key_frames for step, _ in STEPS]
        export_pdf(key_paths, destination / 'key_pairs.pdf')
        manifest.append(dict(sequence=name, directory=folder, title=title, key_source_frames=key_frames, pairs=sequence))
        print(f'{folder}: exported {len(paths)} full PDF pages and {len(key_paths)} key pages', flush=True)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(f'Complete: {args.output}', flush=True)


if __name__ == '__main__':
    main()
