"""Compare terminal gray-point recovery with the preserved hybrid baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from visualize_sparse_hypotheses import caption, draw_points

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PAIRS = ((2433,2435), (2437,2439), (2445,2447), (2463,2465))
STAGES = ('sparse_motion_visualization', 'ownership_stage_comparison',
          'feedback_match_visualization', 'feedback_final_visualization',
          'motion_visualization')


def panel(image, title, detail='', box=None):
    if box is None:
        small = cv2.resize(image, (704,188), interpolation=cv2.INTER_AREA)
    else:
        x0,y0,x1,y1 = box
        crop = image[y0:y1,x0:x1]
        scale = 352/max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_NEAREST)
        small = np.full((352,704,3),255,np.uint8)
        y,x = (352-crop.shape[0])//2, (704-crop.shape[1])//2
        small[y:y+crop.shape[0],x:x+crop.shape[1]] = crop
    return caption(small,title,detail)


def compare(group, baseline):
    records=[]
    comparison_dir=group/'comparisons'
    comparison_dir.mkdir(parents=True,exist_ok=True)
    for first,second in PAIRS:
        new_path=next(group.glob(f'*_{first:010d}_{second:010d}/*/summary.json'))
        old_path=next(baseline.glob(f'*_{first:010d}_{second:010d}/*/summary.json'))
        new_dir,old_dir=new_path.parent,old_path.parent
        new,old=json.loads(new_path.read_text()),json.loads(old_path.read_text())
        cfg=yaml.safe_load((new_dir/'config.yaml').read_text())
        state=np.load(new_dir/'pairwise_state.npz')
        before=np.load(old_dir/'pairwise_state.npz')
        matches=state['anchors_initial']
        labels=state['initial_motion_labels']
        old_labels=before['initial_motion_labels']
        old_count=len(before['H_sparse'])
        assert np.array_equal(matches,before['anchors_initial'])
        assert np.array_equal(state['initial_matches_2d'],before['initial_matches_2d'])
        assert np.array_equal(state['H_sparse'][:old_count],before['H_sparse'])
        assert np.array_equal(labels[old_labels>=0],old_labels[old_labels>=0])
        recovery=new['initialization_report']['gray_recovery']
        recovered=(old_labels<0)&(labels>=0)
        assert int(recovered.sum())==recovery['recovered_points']
        record=dict(frames=[first,second],result=str(new_dir.relative_to(group)),
                    initial_models_before=old_count,initial_models_after=len(state['H_sparse']),
                    initial_supports=new['initial_model_supports'],
                    recovered_points=int(recovered.sum()),
                    gray_before=int((old_labels<0).sum()),gray_after=int((labels<0).sum()),
                    existing_initial_labels_identical=True,existing_initial_models_identical=True,
                    initial_matches_identical=True,
                    final_models_before=old['final_models'],final_models_after=new['final_models'],
                    final_anchors_before=old['final_3d_anchors'],final_anchors_after=new['final_3d_anchors'],
                    recovery=recovery,render={})
        scores=np.load(new_dir/'score_maps.npz')
        old_scores=np.load(old_dir/'score_maps.npz')
        for stage in ('sparse','joint','final'):
            common=(scores[f'coverage_{stage}'] & old_scores[f'coverage_{stage}'] &
                    np.isfinite(scores[stage]) & np.isfinite(old_scores[stage]))
            old_score=float(old_scores[stage][common].mean())
            new_score=float(scores[stage][common].mean())
            record['render'][stage]=dict(common_pixels=int(common.sum()),
                baseline_score=old_score,recovery_score=new_score,
                relative_change_pct=100*(new_score-old_score)/max(old_score,1e-12),
                baseline_coverage=int(old_scores[f'coverage_{stage}'].sum()),
                recovery_coverage=int(scores[f'coverage_{stage}'].sum()))
        data_root=Path(cfg['dataset']['data_root'])
        if not data_root.is_absolute():data_root=ROOT/data_root
        image=cv2.imread(str(data_root/'data_2d_raw'/cfg['dataset']['sequence']/
                              'image_00/data_rect'/f'{first:010d}.png'))
        if image is None:raise FileNotFoundError(first)
        old_draw=draw_points(image,matches[:,:2],old_labels)
        new_draw=draw_points(image,matches[:,:2],labels)
        old_rgb=cv2.cvtColor(old_scores['rendered_rgb_final'],cv2.COLOR_RGB2BGR)
        new_rgb=cv2.cvtColor(scores['rendered_rgb_final'],cv2.COLOR_RGB2BGR)
        rows=[np.hstack((
            panel(old_draw,'Baseline: initial matches',
                  f"K={old_count}; gray={record['gray_before']}"),
            panel(new_draw,'Gray recovery: initial matches',
                  f"K={len(state['H_sparse'])}; recovered={int(recovered.sum())}; gray={record['gray_after']}"))),
              np.hstack((
            panel(old_rgb,'Baseline: final rendering',
                  f"K={old['final_models']}; common score={record['render']['final']['baseline_score']:.5f}"),
            panel(new_rgb,'Gray recovery: final rendering',
                  f"K={new['final_models']}; common score={record['render']['final']['recovery_score']:.5f}")))]
        if first==2463:
            region=((matches[:,0]>=340)&(matches[:,0]<=455)&
                    (matches[:,1]>=165)&(matches[:,1]<=285)&
                    (matches[:,2]-matches[:,0]<-50))
            record['cyclist_diagnostic']=dict(points=int(region.sum()),
                baseline_core=int((old_labels[region]>=0).sum()),
                recovery_core=int((labels[region]>=0).sum()),
                core_by_model=[int((labels[region]==k).sum()) for k in range(len(state['H_sparse']))],
                selection='post-hoc source rectangle and dx < -50 px; never used for fitting')
            rows.append(np.hstack((
                panel(old_draw,'Baseline: cyclist initial matches',box=(330,155,465,295)),
                panel(new_draw,'Recovery: cyclist initial matches',box=(330,155,465,295)))))
            rows.append(np.hstack((
                panel(old_rgb,'Baseline: cyclist final rendering',box=(190,155,385,325)),
                panel(new_rgb,'Recovery: cyclist final rendering',box=(190,155,385,325)))))
        filename=f'{first}_{second}_comparison.png'
        if not cv2.imwrite(str(comparison_dir/filename),np.vstack(rows)):
            raise OSError(filename)
        record['comparison']=f'comparisons/{filename}'
        records.append(record)
    (group/'comparison.json').write_text(json.dumps(records,indent=2))

    lines=['# 初始化末尾：仅恢复未归属匹配的四组实验','',
        '在当前混合几何初始化末尾执行一次恢复，原有合并、竞争和重新拟合保持原流程。'
        '只用最终灰点与合并前候选的支持关系重新拟合旧假设；通过既有核心误差、深度、'
        '6 点支持和解释收益要求后追加候选。已有核心标签、运动矩阵及参考编号在恢复步骤中保持不变，'
        '恢复后不再执行初始化合并。后续联合优化与渲染反馈照常运行。','',
        '合并前支持关系允许重叠，并非每点唯一标签。恢复候选需要明确优于现有运动，'
        '不会直接恢复所有旧标签；仍遵守最多 8 个候选的预算。'
        '本实验只实现末尾恢复，没有修正原先的连锁合并判定。','',
        '对照为留存的 `full_feedback_four_pairs_hybrid_initial`。'
        '四组已逐一验证初始匹配完全相同，原有核心标签与初始运动矩阵完全相同。'
        '该冻结仅指初始化恢复步骤，之后联合优化仍会更新运动和高斯归属。','',
        '| 帧对 | 初始候选 旧→新 | 恢复匹配 | 灰点 旧→新 | 新初始支持数 | 最终候选 旧→新 |',
        '|---|---:|---:|---:|---|---:|']
    for r in records:
        f,s=r['frames']
        lines.append(f"| {f}→{s} | {r['initial_models_before']}→{r['initial_models_after']} | "
            f"{r['recovered_points']} | {r['gray_before']}→{r['gray_after']} | {r['initial_supports']} | "
            f"{r['final_models_before']}→{r['final_models_after']} |")
    lines+=['','## 渲染变化','',
        '以下在两版对应阶段的共同覆盖像素上比较综合渲染残差，越低越好；变化率为新值相对旧值。'
        '全图残差不能单独证明物体归属正确。','',
        '| 帧对 | 初始渲染变化 | 联合优化后变化 | 最终残差 旧→新 | 最终变化 | 最终覆盖像素 旧→新 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in records:
        f,s=r['frames'];v=r['render'];final=v['final']
        lines.append(f"| {f}→{s} | {v['sparse']['relative_change_pct']:+.2f}% | "
            f"{v['joint']['relative_change_pct']:+.2f}% | {final['baseline_score']:.5f}→"
            f"{final['recovery_score']:.5f} | {final['relative_change_pct']:+.2f}% | "
            f"{final['baseline_coverage']}→{final['recovery_coverage']} |")
    cyclist=records[-1]['cyclist_diagnostic']
    lines+=['','## 骑行者与额外背景候选','',
        f"2463→2465 的事后诊断区域有 {cyclist['points']} 对大位移匹配，进入初始核心集的数量由 "
        f"{cyclist['baseline_core']} 变为 {cyclist['recovery_core']}；各新候选对应数量为 "
        f"`{cyclist['core_by_model']}`。该矩形仅用于观察和统计，不是语义真值，也没有参与估计。",'',
        '2463 对比图中，骑行者的独立稀疏运动已经恢复，但最终局部渲染仍有重影和残留。'
        '本组最终保留 6 个候选，恢复步骤同时带回了部分背景近似运动；不能把全图残差下降理解为'
        '骑行者与背景已完全分开。','',
        '恢复也可能重新引入背景近似候选；候选增加、灰点减少不等于物体分割一定更准。'
        '应结合下方初始匹配、联合归属、最终运动和渲染图判断。','',
        '## 关键图像','',
        '| 帧对 | 新旧对比 | 初始候选 | 初始联合优化 | 补匹配 | 反馈前后 | 最终运动 |',
        '|---|---|---|---|---|---|---|']
    for r in records:
        f,s=r['frames'];p=Path(r['result'])
        links=' | '.join(f'[查看]({p/stage/"overview.png"})' for stage in STAGES)
        lines.append(f"| {f}→{s} | [对比]({r['comparison']}) | {links} |")
    lines+=['','每组保留五张阶段 overview，另保留四张对比图，总计 24 张 PNG。'
        '2463 的对比图额外包含骑行者匹配及最终渲染的局部放大。'
        '配置、恢复记录、状态和每轮反馈证据继续保存；未读取动态物体 mask。','',
        '## 复现','',
        '```bash',
        'env CUDA_VISIBLE_DEVICES=3 /media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python '
        'experiments/render_guided_multi_frame/render_guided_two_frame/run_four_pairs.py --config '
        'experiments/render_guided_multi_frame/render_guided_two_frame/configs/kitti360_full_feedback_four_pairs_hybrid_gray_recovery.yaml',
        '/media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python '
        'experiments/render_guided_multi_frame/render_guided_two_frame/compare_gray_recovery.py',
        '```','',
        '开关为 `motion.hybrid.recover_unassigned_after_merge: true`。'
        '默认 `hybrid_initial` 配置未开启该开关，继续执行原留存算法。'
        '源码与配置副本保存在 `source_snapshot/`，检查记录见 `verification.json`。','']
    (group/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ('frames','recovered_points','gray_before','gray_after',
                                      'initial_models_after','final_models_after','render')} for r in records],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group',type=Path,default=HERE/'outputs/full_feedback_four_pairs_hybrid_gray_recovery')
    parser.add_argument('--baseline',type=Path,default=HERE/'outputs/full_feedback_four_pairs_hybrid_initial')
    args=parser.parse_args()
    compare(args.group.resolve(),args.baseline.resolve())
