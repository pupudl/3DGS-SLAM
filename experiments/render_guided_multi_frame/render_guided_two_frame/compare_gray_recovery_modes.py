"""Compare all-unassigned and unexplained-only recovery on identical inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from compare_gray_recovery import HERE, ROOT, PAIRS, STAGES, panel
from visualize_sparse_hypotheses import draw_points


def load_run(group, first, second):
    summary_path = next(group.glob(f'*_{first:010d}_{second:010d}/*/summary.json'))
    run = summary_path.parent
    return (run, json.loads(summary_path.read_text()),
            np.load(run/'pairwise_state.npz'), np.load(run/'score_maps.npz'))


def compare(group, all_group, base_group):
    records=[]
    (group/'comparisons').mkdir(parents=True,exist_ok=True)
    for first,second in PAIRS:
        runs=[load_run(g,first,second) for g in (all_group,group)]
        _,base,base_state,_=load_run(base_group,first,second)
        labels_before=base_state['initial_motion_labels']
        gray=labels_before<0
        cfg=yaml.safe_load((runs[1][0]/'config.yaml').read_text())
        threshold=cfg['motion']['hybrid']['outlier_cost']
        yellow=gray & np.any(base_state['initial_motion_costs']<threshold,axis=1)
        models_before=len(base_state['H_sparse'])
        modes=[]
        for index,(run,summary,state,scores) in enumerate(runs):
            labels=state['initial_motion_labels']
            assert np.array_equal(state['anchors_initial'],base_state['anchors_initial'])
            assert np.array_equal(state['initial_matches_2d'],base_state['initial_matches_2d'])
            assert np.array_equal(state['H_sparse'][:models_before],base_state['H_sparse'])
            assert np.array_equal(labels[~gray],labels_before[~gray])
            recovery=summary['initialization_report']['gray_recovery']
            if index==1:
                assert recovery['mode']=='unexplained_only'
                assert recovery['excluded_yellow_gray_points']==int(yellow.sum())
                assert recovery['eligible_gray_points']==int((gray & ~yellow).sum())
                assert np.all(labels[yellow]<0)
            restored=gray & (labels>=0)
            modes.append(dict(mode='all_unassigned' if index==0 else 'unexplained_only',
                initial_models=len(state['H_sparse']),initial_supports=summary['initial_model_supports'],
                recovered_points=int(restored.sum()),recovered_yellow_gray=int((restored & yellow).sum()),
                remaining_gray=int((labels<0).sum()),final_models=summary['final_models'],
                final_anchors=summary['final_3d_anchors'],recovery=recovery))
        metrics={}
        for stage in ('sparse','joint','final'):
            left,right=runs[0][3],runs[1][3]
            common=(left[f'coverage_{stage}'] & right[f'coverage_{stage}'] &
                    np.isfinite(left[stage]) & np.isfinite(right[stage]))
            a,b=float(left[stage][common].mean()),float(right[stage][common].mean())
            metrics[stage]=dict(common_pixels=int(common.sum()),all_unassigned_score=a,
                unexplained_only_score=b,relative_change_pct=100*(b-a)/max(a,1e-12),
                all_unassigned_coverage=int(left[f'coverage_{stage}'].sum()),
                unexplained_only_coverage=int(right[f'coverage_{stage}'].sum()))
        data_root=Path(cfg['dataset']['data_root'])
        if not data_root.is_absolute():data_root=ROOT/data_root
        image=cv2.imread(str(data_root/'data_2d_raw'/cfg['dataset']['sequence']/
                              'image_00/data_rect'/f'{first:010d}.png'))
        if image is None:raise FileNotFoundError(first)
        draws=[draw_points(image,r[2]['anchors_initial'][:,:2],r[2]['initial_motion_labels']) for r in runs]
        renders=[cv2.cvtColor(r[3]['rendered_rgb_final'],cv2.COLOR_RGB2BGR) for r in runs]
        rows=[np.hstack([panel(draws[k],f"{modes[k]['mode']}: initial matches",
            f"K={modes[k]['initial_models']}; recovered={modes[k]['recovered_points']}; gray={modes[k]['remaining_gray']}")
            for k in range(2)]),np.hstack([panel(renders[k],f"{modes[k]['mode']}: final rendering",
            f"K={modes[k]['final_models']}; common score={metrics['final'][modes[k]['mode']+'_score']:.5f}")
            for k in range(2)])]
        record=dict(frames=[first,second],result=str(runs[1][0].relative_to(group)),
            base_models=models_before,gray_before=int(gray.sum()),yellow_gray_before=int(yellow.sum()),
            strict_eligible_gray=int((gray & ~yellow).sum()),modes=modes,render=metrics,
            same_initial_matches=True,base_models_unchanged=True,base_core_labels_unchanged=True,
            strict_excluded_points_remain_unassigned=True)
        if first==2463:
            matches=base_state['anchors_initial']
            region=((matches[:,0]>=340)&(matches[:,0]<=455)&(matches[:,1]>=165)&
                    (matches[:,1]<=285)&(matches[:,2]-matches[:,0]<-50))
            record['cyclist_diagnostic']=dict(points=int(region.sum()),
                assigned=[int((r[2]['initial_motion_labels'][region]>=0).sum()) for r in runs],
                by_model=[[int((r[2]['initial_motion_labels'][region]==k).sum())
                           for k in range(len(r[2]['H_sparse']))] for r in runs],
                selection='post-hoc source rectangle and dx < -50 px; not an estimation input')
            for pictures,title,box in ((draws,'cyclist initial matches',(330,155,465,295)),
                                       (renders,'cyclist final rendering',(190,155,385,325))):
                rows.append(np.hstack([panel(pictures[k],f"{modes[k]['mode']}: {title}",box=box)
                                       for k in range(2)]))
        filename=f'comparisons/{first}_{second}_modes.png'
        if not cv2.imwrite(str(group/filename),np.vstack(rows)):raise OSError(filename)
        record['comparison']=filename
        records.append(record)
    (group/'mode_comparison.json').write_text(json.dumps(records,indent=2))
    lines=['# 灰点恢复的两种选点模式：四组对比','',
        '模式一 `all_unassigned` 使用全部全局未归属点；模式二 `unexplained_only` 排除其中'
        '被任一恢复前现有候选以混合代价小于阈值解释的点，即某行黄圈对应的全局灰点。'
        '筛选只在恢复开始时计算一次，同时用于拟合和最终接纳，阈值沿用 `outlier_cost`（目前 1）。'
        '不使用缓存旧假设判定黄圈，也不随恢复候选加入而重新定义筛选集合。','',
        '两版输入、随机种子、拟合和验收参数相同，只改变选点模式。已有核心标签和运动矩阵在恢复步骤中保持不变，'
        '之后仍正常联合优化、补点及调整运动结构；本实验没有修正原连锁合并规则。','',
        '| 帧对 | 原灰点 | 模式二排除的黄圈灰点 | 模式二可用点 | 恢复点 模式一→二 | 初始候选 一→二 | 最终候选 一→二 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        a,b=r['modes'];f,s=r['frames']
        lines.append(f"| {f}→{s} | {r['gray_before']} | {r['yellow_gray_before']} | {r['strict_eligible_gray']} | "
            f"{a['recovered_points']}→{b['recovered_points']} | {a['initial_models']}→{b['initial_models']} | "
            f"{a['final_models']}→{b['final_models']} |")
    lines+=['','## 渲染变化','',
        '综合渲染残差在两种模式对应阶段的共同覆盖像素上比较，越低越好。'
        '变化率为模式二相对模式一；全图残差不等同于运动划分准确率。','',
        '| 帧对 | 初始渲染变化 | 联合优化后变化 | 最终残差 一→二 | 最终变化 | 最终覆盖像素 一→二 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in records:
        f,s=r['frames'];v=r['render'];t=v['final']
        lines.append(f"| {f}→{s} | {v['sparse']['relative_change_pct']:+.2f}% | {v['joint']['relative_change_pct']:+.2f}% | "
            f"{t['all_unassigned_score']:.5f}→{t['unexplained_only_score']:.5f} | {t['relative_change_pct']:+.2f}% | "
            f"{t['all_unassigned_coverage']}→{t['unexplained_only_coverage']} |")
    cyclist=records[-1]['cyclist_diagnostic']
    lines+=['','## 骑行者','',
        f"2463→2465 事后诊断区域有 {cyclist['points']} 对大位移匹配，两种模式进入初始核心集的数量为 "
        f"`{cyclist['assigned']}`；各候选的区域内支持分别为 `{cyclist['by_model']}`。"
        '该区域只用于观察统计，不是语义真值，没有参与估计。','',
        f"本组初始候选由模式一的 {records[-1]['modes'][0]['initial_models']} 个变为模式二的 "
        f"{records[-1]['modes'][1]['initial_models']} 个；模式二最终共同覆盖区域残差相对模式一变化 "
        f"{records[-1]['render']['final']['relative_change_pct']:+.2f}%。"
        '两种模式的骑行者最终局部渲染仍有重影，候选减少不代表已经与背景完全分开。','',
        '模式二更保守，可能减少背景近似运动，也可能漏掉被现有候选勉强解释的小运动。'
        '被排除点在本次恢复中保持未归属，后续联合优化仍会使用原有效匹配。','',
        '## 关键图像','',
        '| 帧对 | 两模式对比 | 初始候选 | 联合优化 | 补匹配 | 反馈前后 | 最终运动 |',
        '|---|---|---|---|---|---|---|']
    for r in records:
        f,s=r['frames'];p=Path(r['result'])
        links=' | '.join(f'[查看]({p/stage/"overview.png"})' for stage in STAGES)
        lines.append(f"| {f}→{s} | [对比]({r['comparison']}) | {links} |")
    lines+=['','每组五张阶段 overview，加四张模式对比图，共 24 张 PNG。原模式一结果和源码快照保持留存。',
        '初始化报告记录模式、原灰点数、黄圈灰点数、排除数、实际可用数和恢复明细。','',
        '## 配置与复现','', '```yaml','motion:','  hybrid:',
        '    recover_unassigned_after_merge: true','    gray_recovery_mode: unexplained_only', '```','',
        '切换为 `all_unassigned` 即模式一；不填写新字段也默认模式一。恢复总开关设为 `false` 则关闭两种恢复模式。','',
        '```bash',
        'env CUDA_VISIBLE_DEVICES=3 /media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python '
        'experiments/render_guided_multi_frame/render_guided_two_frame/run_four_pairs.py --config '
        'experiments/render_guided_multi_frame/render_guided_two_frame/configs/kitti360_full_feedback_four_pairs_hybrid_gray_recovery_unexplained.yaml',
        '/media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python experiments/render_guided_multi_frame/render_guided_two_frame/compare_gray_recovery_modes.py',
        '```','', '旧模式四组逐点重放检查见 `mode1_regression.json`；测试与产物检查见 `verification.json`。'
        '源码和配置留存于 `source_snapshot/`。','']
    (group/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ('frames','yellow_gray_before','strict_eligible_gray','render')} for r in records],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group',type=Path,default=HERE/'outputs/full_feedback_four_pairs_hybrid_gray_recovery_unexplained')
    parser.add_argument('--all-group',type=Path,default=HERE/'outputs/full_feedback_four_pairs_hybrid_gray_recovery')
    parser.add_argument('--base-group',type=Path,default=HERE/'outputs/full_feedback_four_pairs_hybrid_initial')
    args=parser.parse_args()
    compare(args.group.resolve(),args.all_group.resolve(),args.base_group.resolve())
