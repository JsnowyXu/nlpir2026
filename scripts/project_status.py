#!/usr/bin/env python
"""Recount verified artifacts; never infer completion from log messages."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import psutil
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from patentflow.data import ROOT, atomic_json, file_hash

protocol_hash = file_hash(ROOT / 'configs/protocol.json')
counts = Counter()
validation = Counter()
deep_validation = Counter()
campaigns = {}
for state in sorted((ROOT / 'state').glob('*campaign*.json')):
    saved = json.loads(state.read_text())
    jobs = saved.get('jobs', {})
    live, stale = [], []
    for key, item in jobs.items():
        if item.get('status') == 'running':
            try:
                process = psutil.Process(item['pid'])
                valid = process.is_running() and process.status() != psutil.STATUS_ZOMBIE
                valid = valid and Path(process.cwd()).resolve() in [ROOT, ROOT/'main_experiment']
            except (psutil.Error, OSError):
                valid = False
            (live if valid else stale).append({'job': key, 'pid': item.get('pid')})
    campaigns[state.stem] = {'jobs': len(jobs), 'states': dict(Counter(j.get('status') for j in jobs.values())),
                             'live_processes': live, 'stale_running_records': stale}
for path in (ROOT / 'results').rglob('metrics.json'):
    try:
        value = json.loads(path.read_text())
    except (ValueError, OSError):
        continue
    if value.get('protocol_sha256') == protocol_hash:
        counts[value.get('method', 'unknown')] += 1
for path in (ROOT / 'results/main').rglob('validation.json'):
    value = json.loads(path.read_text())
    if value.get('protocol_sha256') == protocol_hash:
        validation[value.get('config', {}).get('tag', 'unknown')] += 1
for path in (ROOT / 'results/deep').rglob('validation.json'):
    value = json.loads(path.read_text())
    if value.get('protocol_sha256') == protocol_hash:
        deep_validation[value.get('method', 'unknown')+'/'+value.get('config', {}).get('tag', 'unknown')] += 1
result = {
    'updated': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': protocol_hash,
    'phase': 'three_seed_validation_ablations_and_sensitivity',
    'canonical_main_entrypoint': 'main_experiment/train.py',
    'main_migration_verification': 'reports/main_migration/parity.json',
    'raw_transfer': 'complete_verified',
    'datasets': len(list((ROOT / 'data/processed/views').glob('*.json'))),
    'feasibility_reports': len(list((ROOT / 'reports/feasibility').glob('*.json'))),
    'word2vec_countries': [p.parent.name for p in (ROOT / 'data/text').glob('*/manifest.json')],
    'memory_views': len(list((ROOT / 'data/memory').glob('*/f180_s1095_n1/manifest.json'))),
    'completed_test_runs_by_method': dict(counts),
    'completed_main_validation_runs_by_tag': dict(validation),
    'completed_deep_validation_runs_by_method_and_tag': dict(deep_validation),
    'seeds': [42, 2026, 3407],
    'campaigns': campaigns,
    'pending': ['remaining deep baseline three-seed runs', 'validation-only tuning and final test lock',
                'ablation and sensitivity', 'evolution case studies', 'verified bibliography', 'manuscript and review'],
    'scope': 'Only artifacts matching the current protocol hash are counted. Pilot validation is separate from final test.',
}
atomic_json(ROOT / 'state/progress.json', result)
lines = ['# 实验进度', '', '更新时间（UTC）：'+result['updated'], '',
         '统计仅接受当前协议指纹。验证完成不等于最终测试完成。', '',
         '| 方法 | 验证完成 | 测试完成 | 每阶段目标 |', '|---|---:|---:|---:|']
for method in ['PatentFlow','TOP','PersonalTOP','SVM','XGBoost','RF','EDGPAT','MENCP','RepeatMixer','TGAT','DyGFormer','TAMI','IDGP','TGN','TGN_edgetype']:
    val = validation.get('v2_default',0) if method=='PatentFlow' else deep_validation.get(method+'/v2_default',counts.get(method,0))
    lines.append('| %s | %d | %d | 36 |' % (method,val,counts.get(method,0)))
lines += ['', '## 队列', '', '| 队列 | 完成 | 失败 | 当前存活进程 | 计划任务 |', '|---|---:|---:|---:|---:|']
for name,value in campaigns.items():
    lines.append('| %s | %d | %d | %d | %d |' % (name,value['states'].get('complete',0),value['states'].get('failed',0),len(value['live_processes']),value['jobs']))
lines += ['', '## 主实验独立目录', '',
          '入口 `main_experiment/train.py`；文档 `main_experiment/README.md`；源码 `main_experiment/src/patentflow_main/`。',
          '旧入口仅转发。迁移前后同种子一轮训练、全部排名/逐查询指标及已有checkpoint回放一致，证据 `reports/main_migration/parity.json`。',
          '', '## 仍待完成', '',
          '其余基线三种子、验证选参和最终测试、消融/敏感性完整结果、演化案例与最终向量、参考文献核验、实验后的论文初稿。',
          'TOP/PersonalTOP的三种子输出为确定性重复。迁移验证和pilot不进入正式结果；并行耗时不直接作为无资源争用条件下的效率排名。']
(ROOT/'reports/PROGRESS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
