# 目录整理记录

2026-11-13 按当前运行依赖整理代码。主入口保留为 `run_sequence.py`。

## 当前分类

| 目录 | 内容 |
|---|---|
| `frontend/` | 两帧运动发现、初始归属与运动精修、反馈 |
| `tracking/` | 特征缓存、track 和 object 身份 |
| `ownership/` | 共享高斯窗口、归属优化和空间约束 |
| `mapping/` | 持久地图、参数优化、细化和固定长度子图 |
| `tools/visualization/` | 可重复使用的结果绘图 |
| `tools/validation/` | 运行状态与归档一致性审计 |
| `tools/` | 窗口重放、子图导出 |
| `tests/fixtures/` | 两份仍由回归测试使用的精简配置输入 |

## 清理范围

移动 41 个 Python 文件；删除 30 个旧或一次性 Python 脚本和 30 份旧 YAML。
旧多背景版本备份、版本扫描、固定实验对照、旧运行包装器和一次性诊断不再保留在活动代码中。
原 `profile_sequence.py` 将源码行号写死，无法随代码整理可靠复用，因此也已清理；已测得的报告保留。

各序列配置共用 `configs/kitti360_sequence.yaml`，由它继承 `configs/two_frame.yaml` 的两帧默认值。
序列使用相同的流式窗口、运动身份门控和生命周期，仅输入序列、帧范围和输出目录不同。
2026-11-14 删除只指定旧帧范围的短/长序列预设，保留 drive_0002、drive_0004 两份当前输入配置；
基础地图直接继承共享方法配置，地图参数和原有默认输入范围不变。
旧配置里仍用于回归的参数提取到 `tests/fixtures/`；其余旧版本配置删除。
实验输出保留；早期两帧结果从 `render_guided_two_frame/outputs/` 移至 `outputs/two_frame/`。
历史运行保存的配置、日志和源码快照仍反映运行当时的目录结构；新运行按当前包目录保存完整源码快照。

## 文件迁移

| 原路径 | 现路径 |
|---|---|
| `render_guided_two_frame/hybrid_motion.py` | `frontend/hybrid_motion.py` |
| `render_guided_two_frame/joint_pair.py` | `frontend/joint_pair.py` |
| `render_guided_two_frame/masked_feedback.py` | `frontend/masked_feedback.py` |
| `render_guided_two_frame/pair_utils.py` | `frontend/pair_utils.py` |
| `render_guided_two_frame/strict_two_frame.py` | `frontend/strict_two_frame.py` |
| `feature_tracks.py` | `tracking/feature_tracks.py` |
| `object_state.py` | `tracking/object_state.py` |
| `gaussian_object_backend.py` | `ownership/gaussian_object_backend.py` |
| `object_window.py` | `ownership/object_window.py` |
| `object_render_optimizer.py` | `ownership/object_render_optimizer.py` |
| `object_spatial_constraints.py` | `ownership/object_spatial_constraints.py` |
| `object_height_constraints.py` | `ownership/object_height_constraints.py` |
| `object_connectivity_constraints.py` | `ownership/object_connectivity_constraints.py` |
| `object_masks.py` | `ownership/object_masks.py` |
| `object_gaussian_renderer.py` | `ownership/object_gaussian_renderer.py` |
| `joint_gaussian_visibility.py` | `ownership/joint_gaussian_visibility.py` |
| `persistent_gaussian_map.py` | `mapping/persistent_gaussian_map.py` |
| `gaussian_map_backend.py` | `mapping/gaussian_map_backend.py` |
| `gaussian_map_optimizer.py` | `mapping/gaussian_map_optimizer.py` |
| `gaussian_map_ownership.py` | `mapping/gaussian_map_ownership.py` |
| `gaussian_map_refinement.py` | `mapping/gaussian_map_refinement.py` |
| `gaussian_map_selection.py` | `mapping/gaussian_map_selection.py` |
| `gaussian_submaps.py` | `mapping/gaussian_submaps.py` |
| `visualize_objects.py` | `tools/visualization/visualize_objects.py` |
| `visualize_streaming_objects.py` | `tools/visualization/visualize_streaming_objects.py` |
| `visualize_gaussian_map.py` | `tools/visualization/visualize_gaussian_map.py` |
| `visualize_object_coverage.py` | `tools/visualization/visualize_object_coverage.py` |
| `render_guided_two_frame/visualize_motion.py` | `tools/visualization/visualize_motion.py` |
| `render_guided_two_frame/visualize_sparse_hypotheses.py` | `tools/visualization/visualize_sparse_hypotheses.py` |
| `render_guided_two_frame/visualize_ownership_stages.py` | `tools/visualization/visualize_ownership_stages.py` |
| `render_guided_two_frame/visualize_feedback_matches.py` | `tools/visualization/visualize_feedback_matches.py` |
| `render_guided_two_frame/visualize_feedback_final.py` | `tools/visualization/visualize_feedback_final.py` |
| `verify_full_spatial_run.py` | `tools/validation/verify_full_spatial_run.py` |
| `verify_streaming_run.py` | `tools/validation/verify_streaming_run.py` |
| `verify_gaussian_map_run.py` | `tools/validation/verify_gaussian_map_run.py` |
| `verify_gaussian_submaps.py` | `tools/validation/verify_gaussian_submaps.py` |
| `replay_object_window.py` | `tools/replay_object_window.py` |
| `export_gaussian_submaps.py` | `tools/export_gaussian_submaps.py` |
| `render_guided_two_frame/test_hybrid_motion.py` | `tests/test_hybrid_motion.py` |
| `render_guided_two_frame/test_gray_recovery.py` | `tests/test_gray_recovery.py` |
| `render_guided_two_frame/test_masked_feedback.py` | `tests/test_masked_feedback.py` |

## 删除的脚本与配置

```text
compare_spatial_constraints.py
configs/kitti360_spatial_full_recovered.yaml
diagnose_motion_identity.py
diagnostics/kitti360_joint_map_refined_20261110_000907/compare_saved_runs.py
diagnostics/submaps_implementation_validation/lsgslam_submap_gpu_budget_smoke.py
diagnostics/submaps_implementation_validation/run_lsgslam_submap_cuda_boundary_smoke.py
profile_sequence.py
render_guided_two_frame/analyze_motion_partition.py
render_guided_two_frame/backups/hybrid_multibg_early/compare_versions.py
render_guided_two_frame/backups/hybrid_multibg_early/configs/kitti360_full_feedback_four_pairs_hybrid_multibg_early.yaml
render_guided_two_frame/backups/hybrid_multibg_early/hybrid_motion.py
render_guided_two_frame/backups/hybrid_multibg_early/joint_pair.py
render_guided_two_frame/backups/hybrid_multibg_early/masked_feedback.py
render_guided_two_frame/backups/hybrid_multibg_early/pair_utils.py
render_guided_two_frame/backups/hybrid_multibg_early/run_four_pairs.py
render_guided_two_frame/backups/hybrid_multibg_early/strict_two_frame.py
render_guided_two_frame/backups/hybrid_multibg_early/visualize_feedback_final.py
render_guided_two_frame/backups/hybrid_multibg_early/visualize_feedback_matches.py
render_guided_two_frame/backups/hybrid_multibg_early/visualize_motion.py
render_guided_two_frame/backups/hybrid_multibg_early/visualize_ownership_stages.py
render_guided_two_frame/backups/hybrid_multibg_early/visualize_sparse_hypotheses.py
render_guided_two_frame/cleanup_outputs.py
render_guided_two_frame/compare_gray_recovery.py
render_guided_two_frame/compare_gray_recovery_modes.py
render_guided_two_frame/compare_masked_feedback.py
render_guided_two_frame/compare_sparse_ransac_sampling.py
render_guided_two_frame/configs/kitti360_0000.yaml
render_guided_two_frame/configs/kitti360_2433_2435_balanced.yaml
render_guided_two_frame/configs/kitti360_2433_2435_high_recall.yaml
render_guided_two_frame/configs/kitti360_anchor_gaussian.yaml
render_guided_two_frame/configs/kitti360_feedback_prune_diagnostic.yaml
render_guided_two_frame/configs/kitti360_feedback_v10.yaml
render_guided_two_frame/configs/kitti360_feedback_v11.yaml
render_guided_two_frame/configs/kitti360_feedback_v13.yaml
render_guided_two_frame/configs/kitti360_feedback_v2.yaml
render_guided_two_frame/configs/kitti360_feedback_v2_conservative.yaml
render_guided_two_frame/configs/kitti360_feedback_v3.yaml
render_guided_two_frame/configs/kitti360_feedback_v4.yaml
render_guided_two_frame/configs/kitti360_feedback_v5.yaml
render_guided_two_frame/configs/kitti360_feedback_v6.yaml
render_guided_two_frame/configs/kitti360_feedback_v7.yaml
render_guided_two_frame/configs/kitti360_feedback_v8.yaml
render_guided_two_frame/configs/kitti360_feedback_v9.yaml
render_guided_two_frame/configs/kitti360_full_feedback_four_pairs.yaml
render_guided_two_frame/configs/kitti360_full_feedback_four_pairs_hybrid_gray_recovery.yaml
render_guided_two_frame/configs/kitti360_full_feedback_four_pairs_hybrid_gray_recovery_unexplained.yaml
render_guided_two_frame/configs/kitti360_full_feedback_four_pairs_hybrid_initial.yaml
render_guided_two_frame/configs/kitti360_full_feedback_four_pairs_ransac8.yaml
render_guided_two_frame/configs/kitti360_full_feedback_four_pairs_ransac8_depth5_50.yaml
render_guided_two_frame/configs/kitti360_full_feedback_four_pairs_ransac8_depth_fit_global_mask_repair.yaml
render_guided_two_frame/configs/kitti360_full_feedback_four_pairs_ransac8_depth_fit_masked_repair.yaml
render_guided_two_frame/configs/kitti360_full_feedback_four_pairs_ransac8_depth_fit_score_mask.yaml
render_guided_two_frame/configs/kitti360_full_feedback_four_pairs_ransac8_depth_fit_weighted.yaml
render_guided_two_frame/configs/kitti360_full_feedback_four_pairs_ransac8_depth_weighted.yaml
render_guided_two_frame/evaluate_unmasked_pair.py
render_guided_two_frame/evaluate_validation.py
render_guided_two_frame/run_experiment.py
render_guided_two_frame/run_four_pairs.py
render_guided_two_frame/run_validation.py
summarize_window_run.py
run_spatial_full_recovered.sh
```

整理前的代码、配置和说明备份在 `/tmp/rgmf-before-cleanup-20261113.tar.gz`。
备份只用于本次整理回退，不放回实验目录。

## 验证

整理前：多帧回归 328 项通过（CPU 环境跳过 13 项 CUDA 检查），两帧回归 45 项通过。
整理后：原两套测试合并为同一入口；最终测试与配置等价性记录见 [整理验证](cleanup_validation.json)。
