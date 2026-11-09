"""Build a complete, credential-screened journal from actual B601 run artifacts."""
import csv
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
from urllib.parse import parse_qsl, urlsplit
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parent
KIT=ROOT.parent/'rebot-b601-training/openpi_b601'
ASSESS=ROOT.parent/'B601_SERVER_ASSESSMENT_20261109'
OUT=ROOT/'EXPERIMENT_RECORDS_20261110'
MON=KIT/'runs/pi05_b601_lora/b601-final60-v1_monitor'
OUT.mkdir(exist_ok=False)
SOURCES={};REDACTIONS=[];EXCLUDED=[]
TEXT={'.log','.txt','.md','.json','.jsonl','.csv','.py','.sh','.toml','.lock','.patch','.svg'}
SAFE={*TEXT,'.npz','.png','.jpg','.jpeg','.pdf'}
FORBIDDEN={'.git','.ssh','.env','.venv','.openpi','__pycache__','token','stored_tokens','credentials'}
PATTERNS={
 'hf_token':re.compile(r'\bhf_[A-Za-z0-9]{20,}\b'),
 'github_token':re.compile(r'\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b'),
 'github_fine_grained_token':re.compile(r'\bgithub_pat_[A-Za-z0-9_]{30,}\b'),
 'other_api_secret':re.compile(r'\bsk-[A-Za-z0-9_-]{24,}\b'),
 'aws_access_key':re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
 'private_key':re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----'),
 'authorization_header':re.compile(r'(?i)(authorization\s*[:=]\s*["\x27]?(?:bearer|token)\s+)[A-Za-z0-9._~+/=-]{16,}'),
 'credential_url':re.compile(r'https?://[^\s/:@]+:[^\s/@]+@'),
}
URL=re.compile(r'https?://[^\s"\x27<>]+')
SIGNED_KEYS={'signature','policy','key-pair-id','token','access_token','x-amz-signature','x-amz-credential','x-amz-security-token','x-goog-signature','x-goog-credential'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def utc8(value):return datetime.datetime.fromtimestamp(value,ZoneInfo('Asia/Shanghai')).isoformat()
def sanitized(data,relative):
    value=data.decode('utf-8',errors='strict');changes=[]
    for name,pattern in PATTERNS.items():
        value,count=pattern.subn('[CREDENTIAL_REDACTED]',value)
        if count:changes.append({'pattern':name,'count':count})
    def url_replacement(match):
        raw=match.group(0)
        try:parsed=urlsplit(raw);keys={k.lower() for k,v in parse_qsl(parsed.query)}
        except ValueError:return raw
        if parsed.query and keys & SIGNED_KEYS:
            changes.append({'pattern':'signed_url_query','count':1})
            return raw.partition('?')[0]+'?[SIGNED_QUERY_REDACTED]'
        return raw
    value=URL.sub(url_replacement,value)
    if changes:REDACTIONS.append({'file':relative,'redactions':changes})
    return value.encode('utf-8')
def add(source,relative,category):
    source=Path(source);relative=str(relative)
    assert not any(p in FORBIDDEN for p in Path(relative).parts),relative
    if not source.is_file():return
    assert source.stat().st_size<90*1024*1024,(source,source.stat().st_size)
    dest=OUT/relative;assert not dest.exists(),relative;dest.parent.mkdir(parents=True,exist_ok=True)
    data=source.read_bytes();original=hashlib.sha256(data).hexdigest()
    if source.suffix.lower() in TEXT or source.name.startswith('LICENSE') or source.name=='NOTICE':data=sanitized(data,relative)
    dest.write_bytes(data)
    SOURCES[relative]={'source_path':str(source),'source_bytes':source.stat().st_size,'source_sha256':original,
                       'published_bytes':len(data),'published_sha256':hashlib.sha256(data).hexdigest(),
                       'byte_identical_to_source':original==hashlib.sha256(data).hexdigest(),'category':category}
def tree(source,prefix,category):
    for p in sorted(Path(source).rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(source)
        if any(x in FORBIDDEN or x in ('checkpoint','params','train_state','cache') for x in rel.parts):continue
        if p.suffix.lower() not in SAFE or p.stat().st_size>=90*1024*1024:
            EXCLUDED.append({'source_path':str(p),'bytes':p.stat().st_size,'reason':'not a supported record file or exceeds journal file-size limit'});continue
        add(p,Path(prefix)/rel,category)
def write_json(relative,value):
    p=OUT/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


# Every stage log and every receipt, including failed first attempts and ACT comparison.
tree(ROOT/'logs','raw_logs','original stage stdout/stderr')
tree(ROOT/'receipts','stage_receipts','original stage commands, timing, exit code and acceptance')
tree(ROOT/'diagnostics','diagnostics','acceptance diagnostics and rejected probes')
tree(ROOT/'evaluation','evaluation','canonical final action evaluation')
tree(ROOT/'early_evaluation','early_evaluation','earlier noncanonical diagnostic evaluation')
tree(ROOT/'source_versions','source_snapshots/executed_history','executed/archive source history')
tree(ROOT/'LOSS_AUDIT_20261110','loss_audit','post-run loss audit; no new model training')
for p in sorted(MON.glob('*')):
    if p.is_file() and p.suffix in ('.json','.jsonl'):add(p,Path('monitor')/p.name,'formal checkpoint and validation monitor')
for name in ('RUN_CONTRACT.json','ENVIRONMENT.json','DATA_ACCEPTANCE.json','JAX_GPU_ACCEPTANCE.json',
             'TRAINING_SOURCE_LOCK.json','PRE_SMOKE_SOURCE_HASHES.json','INFERENCE_SOURCE_LOCK.json',
             'FINAL_ACCEPTANCE_SOURCE_LOCK.json','LR_SCHEDULE_CHECK.json','ACTION_SELECTION_PROTOCOL.json',
             'ACTION_SELECTION_RESULT.json','DEPLOYMENT_TARGET.json','ISSUE_RESOLUTION_LOG.json',
             'WAITING_CONTROLLER_PI05_PRIORITY_RELOAD.json'):
    add(ROOT/name,Path('configuration')/name,'protocol, environment or decision snapshot')
for p in sorted(ROOT.glob('QUALITY_REVIEW_*.json')):add(p,Path('configuration')/p.name,'runtime model-quality review')
for name in ('CONSOLE_TRAIN_METRICS.jsonl','CONSOLE_WATCH_OFFSET.json','GPU_SAMPLES.csv',
             'WAIT_OBSERVATIONS.jsonl','DELIVERY_OBSERVATIONS.jsonl','HF_PUBLICATION_OBSERVATIONS.jsonl'):
    add(ROOT/name,Path('resources')/name,'sampled metrics/process observations')
for p in sorted(ROOT.glob('WAIT_OBSERVATION_*.json')):add(p,Path('resources')/p.name,'point-in-time runtime observation')
for name in ('INDEPENDENT_FINAL_AUDIT.json','ROOT_COMPLETION_AUDIT.json','FINAL_GOAL_COMPLETION_AUDIT.json',
             'DELIVERY_RECEIPT.json','FINAL_ACCEPTANCE_STATUS.json','PIPELINE_STATUS.json',
             'B601_PI05_DEPLOYMENT.sha256','B601_PI05_WINDOWS_CLIENT.sha256'):
    add(ROOT/name,Path('audit_receipts')/name,'completed server validation and delivery')
for name in ('HF_PUBLICATION_RECEIPT.json','HF_PUBLICATION_VERIFIED.json','GITHUB_PUBLICATION_RECEIPT.json',
             'GITHUB_PUBLICATION_VERIFIED.json','GITHUB_MODEL_LINK_RECEIPT.json'):
    add(ROOT/name,Path('publication_evidence')/name,'verified private publication')
for name in ('HF_PRIVATE_VISIBILITY_RECEIPT.json','VERBATIM_LICENSE_WHITESPACE_EXCEPTION.json'):
    add(ROOT/'PUBLICATION_PREP'/name,Path('publication_evidence')/name,'publication issue resolution')
for name in ('ASSESSMENT_MANIFEST.json','PREFLIGHT_EVIDENCE.json','SERVER_PLAN_CN.md','run_on_server.sh','b601_server_adaptation.patch'):
    add(ASSESS/name,Path('hardware_and_plan')/name,'pre-run server evidence and approved adaptation')
tree(ASSESS/'proposed','source_snapshots/server_proposed','pre-run proposed adaptation')
tree(ASSESS/'upstream_source','source_snapshots/b601_upstream','original assignment sources')
for name in ('b601_data.py','train_pi05.py','longrun_monitor.py','b601_policy.py','evaluate_b601.py','serve_b601.py','prepare_dataset.py','run_pi05.sh','README.md'):
    add(KIT/name,Path('source_snapshots/current_b601')/name,'current source; execution versions separately archived')
for name in ('pyproject.toml','uv.lock','LICENSE','LICENSE_GEMMA.txt'):
    add(KIT/'.openpi'/name,Path('source_snapshots/openpi_pinned')/name,'pinned OpenPI dependency/source provenance')
for name in ('scripts/train.py','src/openpi/models/pi0.py','src/openpi/models/model.py',
             'src/openpi/training/config.py','src/openpi/transforms.py','src/openpi/training/data_loader.py',
             'src/openpi/training/checkpoints.py','src/openpi/training/weight_loaders.py','src/openpi/shared/download.py'):
    add(KIT/'.openpi'/name,Path('source_snapshots/openpi_pinned')/name,'pinned OpenPI implementation used by this run')
for p in sorted(ROOT.glob('*.py')):add(p,Path('source_snapshots/server_tools')/p.name,'server orchestration, analysis and verification tools')
for name in ('executed_b601_changes.patch','executed_openpi_download_fix.patch'):
    add(ROOT/name,Path('source_snapshots')/name,'saved source patch')
for name in ('splits.json','DATASET.sha256','episodes.csv'):
    add(KIT.parent/name,Path('data_and_base/original_repository')/name,'original final60 split/checksum metadata')
add(KIT/'prepared/manifest.json','data_and_base/converted_manifest.json','actual converted dataset split binding')
for split in ('rebot-b601-train','rebot-b601-val'):
    for name in ('info.json','episodes.jsonl','tasks.jsonl','stats.json'):
        add(KIT/'prepared/datasets'/split/'meta'/name,Path('data_and_base')/split/name,'converted LeRobot metadata')
add('/data2/yujiegao/b601-pi05/models/pi05_base/VERIFIED_DOWNLOAD.json','data_and_base/base_verified_download.json','official base object generation/checksum proof')
add(ROOT/'DATA_CAMERA_CONTACT_SHEET.jpg','data_and_base/DATA_CAMERA_CONTACT_SHEET.jpg','sampled real train/validation observations')
# Independent ACT source/summary only. Its weights/cache stay with its own artifact.
act=ROOT.parent/'B601_ACT_20261109'
for p in sorted(act.glob('*')):
    if p.is_file() and p.suffix in ('.py','.json','.md','.patch','.sh'):
        add(p,Path('auxiliary_act')/p.name,'independent ACT comparison; not pi05 conversion')

stages=[]
for p in sorted((ROOT/'receipts').glob('*.json')):
    value=read(p)
    if 'started_at' not in value or 'command' not in value:continue
    stages.append({'label':value.get('label',p.stem),'started_at':value['started_at'],
        'finished_at':value.get('finished_at',''),'status':value.get('status',''),
        'returncode':value.get('returncode',''),'elapsed_seconds':value.get('elapsed_seconds',''),
        'command_json':json.dumps(value['command'],ensure_ascii=False),
        'log':'raw_logs/'+Path(value['log']).name,'receipt':'stage_receipts/'+p.name})
stages.sort(key=lambda x:x['started_at'])
with (OUT/'TIMELINE.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(stages[0]));writer.writeheader();writer.writerows(stages)
table=['# 全部程序阶段与退出结果\n','命令完整保存在对应receipt和TIMELINE.csv；时间为带时区的原始记录。\n',
       '| 阶段 | 开始 | 耗时秒 | 状态/退出码 | 日志 |','| --- | --- | ---: | --- | --- |']
for s in stages:table.append(f"| {s['label']} | {s['started_at']} | {s['elapsed_seconds']} | {s['status']} / {s['returncode']} | [log]({s['log']}) / [receipt]({s['receipt']}) |")
(OUT/'TIMELINE_CN.md').write_text('\n'.join(table)+'\n')

gpu=list(csv.DictReader((ROOT/'GPU_SAMPLES.csv').open(newline='')))
session=read(MON/'SESSION_START.json');completion=read(MON/'LONGRUN_COMPLETE.json')
begin=datetime.datetime.fromisoformat(session['first_dispatch_completed_at']).timestamp()
end=datetime.datetime.fromisoformat(completion['at']).timestamp()
resource={'raw_rows':len(gpu),'recording_first_at':utc8(float(gpu[0]['unix_time'])),
          'recording_last_at':utc8(float(gpu[-1]['unix_time'])),'nominal_sampler_sleep_seconds':1,
          'timing_note':'actual period includes nvidia-smi query time; sampled max is not exact physical peak',
          'formal_first_dispatch_at':session['first_dispatch_completed_at'],'formal_completed_at':completion['at'],'gpus':{}}
for index in ('0','1','2','3'):
    rows=[r for r in gpu if r['gpu_index']==index and begin<=float(r['unix_time'])<=end]
    times=[float(r['unix_time']) for r in rows]
    resource['gpus'][index]={'formal_samples':len(rows),'median_period_seconds':statistics.median(b-a for a,b in zip(times,times[1:])),
        'max_gap_seconds':max(b-a for a,b in zip(times,times[1:])),
        'statistics':{key:{'mean':statistics.mean(float(x[key]) for x in rows),'max':max(float(x[key]) for x in rows)}
                      for key in ('memory_used_MiB','gpu_utilization_percent','temperature_C','power_W')}}
write_json('resources/RESOURCE_SUMMARY.json',resource)
summary=['# GPU资源采样复核\n',f"原始CSV共{len(gpu):,}行（4张卡各42,236条），覆盖{resource['recording_first_at']}至{resource['recording_last_at']}。正式训练按第一次成功dispatch到最终checkpoint时间截取。采样器每轮sleep1秒，实际中位间隔约1.110秒，最大间隔约1.310秒。\n",
         '| GPU | 正式区间样本 | 平均利用率% | 观测最大显存MiB | 平均温度℃ | 最大温度℃ | 平均功耗W |',
         '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for index in ('0','1','2','3'):
    r=resource['gpus'][index];s=r['statistics']
    summary.append(f"| {index} | {r['formal_samples']} | {s['gpu_utilization_percent']['mean']:.3f} | {s['memory_used_MiB']['max']:.0f} | {s['temperature_C']['mean']:.2f} | {s['temperature_C']['max']:.0f} | {s['power_W']['mean']:.2f} |")
summary.append('\nπ₀.₅正式FSDP训练使用GPU2/3；GPU1有独立ACT和推理诊断，GPU0主要为桌面。显存包含分配池，采样可能漏短峰值。CPU/系统内存为保存的快照和诊断结果，不声称有连续CPU内存全程遥测。')
(OUT/'resources/RESOURCE_SUMMARY_CN.md').write_text('\n'.join(summary)+'\n')

# Keep original terminal bytes, and also provide a reader-friendly formal log.
ansi=re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
formal=(OUT/'raw_logs/formal-longrun-v1.log').read_text(errors='replace')
(OUT/'FORMAL_LOG_READABLE.txt').write_text(ansi.sub('',formal).replace('\r','\n'))

overview=f'''# 服务器B601完整实验记录

本目录提供2026-11-09至2026-11-10的实际服务器执行证据：环境与数据准备、两组20步短跑、12.273248小时正式训练、43,001次优化器迭代、完整离线评估、恢复/部署检查、异常与修复，以及后验loss复核。

## 阅读顺序

1. [最终方法、硬件与结论](../docs/server_final60/FINAL_REPORT_CN.md)
2. [本次loss曲线复核](loss_audit/LOSS_AUDIT_CN.md)：训练loss总体下降；验证flow在5k后回升；实际动作指标比较和局限分开说明。
3. [全部阶段时间线](TIMELINE_CN.md) / [可下载CSV](TIMELINE.csv)：包含实际命令、开始/结束时间、退出码，成功和失败尝试都保留。
4. [原始正式训练日志](raw_logs/formal-longrun-v1.log) / [去终端控制符的阅读版](FORMAL_LOG_READABLE.txt)
5. [完整训练曲线CSV](loss_audit/training_curve.csv)、[43次验证flow](loss_audit/validation_flow_curve.csv)、[逐episode动作比较](loss_audit/per_episode_comparison.csv)
6. [GPU资源采样说明](resources/RESOURCE_SUMMARY_CN.md) / [全部168,944行GPU原始CSV](resources/GPU_SAMPLES.csv)
7. [失败原因和修复记录](configuration/ISSUE_RESOLUTION_LOG.json)、[最终独立审计](audit_receipts/INDEPENDENT_FINAL_AUDIT.json)
8. [下载与部署指南](../docs/server_final60/DOWNLOAD_AND_START_CN.md)

![loss与实际动作误差](loss_audit/loss_and_action_curves.png)

## 实验参数与数据

- Linux Lenovo ThinkSystem ST650 V3，双Xeon Gold6530，Linux可见内存约503GiB，4张RTX A4500 20GB；正式π₀.₅使用物理GPU2/3、FSDP2、global batch4、workers2、seed42。
- OpenPI固定commit215abfb217dbac7d5f1273282331b9b1866c0479，JAX/JAXlib0.5.3。LoRA语言/动作分支，原生freeze filter同时训练视觉参数；完整环境见[ENVIRONMENT](configuration/ENVIRONMENT.json)，精确实际参数见[RUN_PARAMETERS](monitor/RUN_PARAMETERS.json)。
- final60固定50训练/10验证，28,729/5,118帧，名义25fps、两路RGB、7轴原始反馈/absolute_adapter_input动作、horizon25。训练统计用于归一化，验证复用同一统计。
- 原始数据见[私有final-60 Release](https://github.com/ken-ab/rebot-b601-training/releases/tag/final-60)，分割/校验/转换信息见[data_and_base](data_and_base/)。官方base下载的20个对象及校验见[base_verified_download](data_and_base/base_verified_download.json)。
- 20步smoke分别batch2和4，原始20条loss/梯度与checkpoint读回见[batch2](raw_logs/smoke-b2-v1.log)、[batch4](raw_logs/smoke-b4-v1.log)及对应stage_receipts。
- 正式训练最大200k为保护上限；LR warmup500、peak2.5e-5、decay10k到2.5e-6，save1000，按12.25小时后的完整保存点结束。实际活动时间{completion['active_seconds']:.6f}秒，末编号43000、优化器计数43001。
- [SESSION_START](monitor/SESSION_START.json)、[LONGRUN_COMPLETE](monitor/LONGRUN_COMPLETE.json)与[最后checkpoint参数/优化器读回](diagnostics/final_optimizer.json)相互绑定。计时排除准备和初始编译，包含正常验证与保存。

## 原始日志索引

| 阶段 | 主要实际记录 |
| --- | --- |
| 环境/转换 | [prepare-v1失败](raw_logs/prepare-v1.log)、[prepare-v2通过](raw_logs/prepare-v2.log)、[60条episode数据审计](raw_logs/full-data-audit-v1.log) |
| 基座下载与GPU | [首次校验处理失败](raw_logs/base-download-v1.log)、[修复后下载校验](raw_logs/base-download-v2.log)、[双GPU集合通信](raw_logs/jax-gpu-collective-v1.log) |
| 真实20步短跑 | [batch2](raw_logs/smoke-b2-v1.log)、[batch4](raw_logs/smoke-b4-v1.log)、[batch4完整参数读回](stage_receipts/smoke-b4-v1-orbax-readback.json) |
| 保存与恢复探针 | [首次探针](raw_logs/hook-probe-v1.log)、[从checkpoint10恢复并保存20](raw_logs/hook-resume-v1.log) |
| 正式长跑 | [完整原始stdout/stderr](raw_logs/formal-longrun-v1.log)、[431条训练记录](resources/CONSOLE_TRAIN_METRICS.jsonl)、[43个验证保存点](monitor/VALIDATION_METRICS.jsonl) |
| 真实动作评估 | [纯基座](raw_logs/full-eval-base-v1.log)、[flow最优5k](raw_logs/full-eval-best-v1.log)、[10k确定性复核](raw_logs/repro-candidate-checkpoint10000-v1.log)、[末端43k](raw_logs/full-eval-last-v1.log) |
| 数值一致性 | [严格自检原失败](raw_logs/deployment-rehearsal10000-v1.log)、[固定选核后通过](raw_logs/deployment-rehearsal10000-v2.log)、[问题与协议](diagnostics/DETERMINISM_V2_ACCEPTANCE.json) |
| 网络与客户端 | [紧凑RPC原失败](raw_logs/websocket-compact-rehearsal-v1.log)、[显式直连后通过](raw_logs/websocket-compact-rehearsal-v2.log)、[最终RPC](raw_logs/websocket-final-v1.log)、[211观察像素/线格式等价](diagnostics/CLIENT_PREPROCESS_EQUIVALENCE.json) |
| 8GB路线与部署检查 | [小BFC池OOM](raw_logs/inference-pool-7424-v1.log)、[最终platform分配器](raw_logs/final-low-memory-profile-v1.log)、[动作检查器36案例](raw_logs/action-guard-acceptance-v2.log) |
| 最终交付 | [模型包导出](raw_logs/deployment-export-v1.log)、[严格模型读回](raw_logs/deployment-inference-readback-v1.log)、[优化器读回](raw_logs/final-optimizer-readback-v1.log)、[独立归档审计](raw_logs/independent-final-delivery-audit-v1.log) |
| 独立ACT对照 | [ACT训练日志](raw_logs/act-formal-training-v1.log)、[ACT配置与结果](auxiliary_act/)；它不是π₀.₅转换版，未替换主模型 |
| 私有发布 | [HF上传日志](raw_logs/hf-publication-v1.log)、[全部138文件核验回执](publication_evidence/HF_PUBLICATION_RECEIPT.json)、[固定下载版本写入GitHub](raw_logs/github-model-link-v1.log) |

## 指标、版本与覆盖边界

训练loss是100步区间均值、四位小数；step0例外为单步。本次W&B禁用重绑定曾让计划中的单步hook失效，外部监测器从原始stdout恢复了431条记录。未保存的43,001条逐步全精度loss不能事后补造。

`observed_at`表示采集器读到日志的时间，启动时可能一次收集多条旧记录；不是每一步的精确GPU结束时间。训练计时以SESSION_START和checkpoint绑定时间为准。实时状态观察可能重复同一step，不能当作额外训练数据点。

`validation_flow_loss`对固定80观察的归一化32维向量场求MSE，含25补齐维与12%重复尾步；它不等于真实7轴动作误差。完整动作评估使用固定seed、每个观察单次采样、10步去噪积分，在211窗口中计5,118有效动作行、排除157尾padding行，分别报告六关节平均MAE和夹爪MAE。详细源代码和复算CSV在loss_audit。

`BEST.json`保存的是5k的flow代理最优点。最终选模在预定最多4个distinct候选中比较真实关节/夹爪MAE，实际有5k、10k、43k三个不同候选，选中43k。`best/candidate_flow_best`、`last/candidate_last`是同一候选的不同来源标签，不是额外独立模型。

[evaluation](evaluation/)是统一固定推理flags下的正式结果；[early_evaluation](early_evaluation/)是最初自动选核配置下的诊断/被替代结果，保留追踪，不与正式结果混成同一协议。跨进程原golden差异保留，固定推理XLA flags后重做候选；训练本身没有因此重启。strict容限未放宽。

[source_snapshots/executed_history](source_snapshots/executed_history/)含正式训练开始时实际源码与后续补救前版本；[current_b601](source_snapshots/current_b601/)是修复后的当前文件。当前文件不能冒充全部12小时执行版本，TRAINING_SOURCE_LOCK与历史目录用于确定执行来源。优化器/参数/计数恢复已实际测试；上游DataLoader位置不恢复，不承诺逐步完全一致续训。

完整权重和原始视频分别位于[HF固定私有模型版本](https://huggingface.co/dahjgfa/b601-pi05-final60/tree/8c11eb8751d5cc6f688fa51db523da5bd8f2518a)与原私有Release。本目录集中保存日志、参数配置、数据/权重校验、预测、监控和源码；凭据文件、令牌缓存和私人会话记录不属于实验归档。

## 如何核验

[MANIFEST.json](MANIFEST.json)记录每个归档文件的源路径、源SHA256及发布SHA256；发生凭据脱敏时明确标记，原服务器文件保持原样。[SCAN_REPORT.json](SCAN_REPORT.json)只记录命中文件/模式/数量，不记录秘密值。Git属性保证原始日志、CSV和源码快照不会被换行转换；原始文本格式不做代码风格修改。SHA256SUMS包含MANIFEST、不含自身。

```bash
sha256sum -c SHA256SUMS
```

GitHub对较大日志/CSV的在线预览可能不完整，下载原文件可获得全部内容。复核脚本为[source_snapshots/server_tools/audit_loss_curves.py](source_snapshots/server_tools/audit_loss_curves.py)，包含服务器路径，异机重放需适配数据位置。图表、CSV及JSON均由实际日志/已保存预测生成；本轮复核没有重新训练或替换已发布模型。

本次证据支持离线示教动作适配有效；验证集参与选模，只有10条验证episode，没有多种子和独立测试集；Windows4070/WSL、相机/USB/原适配器及机械臂闭环仍需现场联调，软件检查不能物理停止已发出的动作。
'''
(OUT/'README.md').write_text(overview)

verbatim_dirs=['raw_logs','stage_receipts','monitor','resources','configuration','audit_receipts',
    'publication_evidence','hardware_and_plan','data_and_base','diagnostics','evaluation','early_evaluation',
    'source_snapshots','auxiliary_act']
attrs=['# Preserve source bytes in the evidence archive; curated Markdown remains normal text.']
attrs += [d+'/** -text -whitespace' for d in verbatim_dirs]
attrs += ['TIMELINE.csv -text -whitespace','FORMAL_LOG_READABLE.txt -text -whitespace',
          'loss_audit/*.csv -text -whitespace','loss_audit/*.json -text -whitespace',
          'loss_audit/*.svg -text -whitespace','loss_audit/*.pdf -text','loss_audit/*.png -text']
(OUT/'.gitattributes').write_text('\n'.join(attrs)+'\n')

# Fail closed if anything recognizable remains; values are never logged.
findings=[];text_files=0
for p in OUT.rglob('*'):
    if not p.is_file():continue
    if p.suffix.lower() in TEXT or p.name in ('README.md','.gitattributes') or p.name.startswith('LICENSE'):
        value=p.read_text(errors='strict');text_files+=1
        for name,pattern in PATTERNS.items():
            if pattern.search(value):findings.append({'file':str(p.relative_to(OUT)),'pattern':name})
assert not findings,findings
coverage={'generated_at':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    'stage_logs':len(list((OUT/'raw_logs').glob('*'))),'stage_receipts':len(list((OUT/'stage_receipts').glob('*'))),
    'actual_program_stages_in_timeline':len(stages),'training_metric_rows':431,'validation_checkpoint_rows':43,
    'gpu_rows':len(gpu),'final_active_hours':completion['active_seconds']/3600,'optimizer_updates':43001,
    'text_files_scanned':text_files,'credential_findings_after_redaction':findings,
    'redactions':REDACTIONS,'excluded_record_candidates':EXCLUDED,
    'scope':'All retained B601 stage logs/receipts, protocol/resources, canonical+rejected evaluations, source history and post-run audit; no credentials or model binaries'}
write_json('SCAN_REPORT.json',coverage)
inventory={}
for p in sorted(OUT.rglob('*')):
    if not p.is_file():continue
    rel=str(p.relative_to(OUT));entry={'bytes':p.stat().st_size,'sha256':sha(p)}
    if rel in SOURCES:entry.update(SOURCES[rel])
    else:entry.update({'source_path':None,'category':'generated post-run archive index/statistic'})
    inventory[rel]=entry
write_json('MANIFEST.json',{'schema_version':1,'coverage':coverage,'files':inventory})
(OUT/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(OUT))+'\n' for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
receipt={'status':'prepared_not_uploaded','output':str(OUT),'files':len(inventory)+2,
         'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),
         'manifest_sha256':sha(OUT/'MANIFEST.json'),'coverage':coverage}
(ROOT/'EXPERIMENT_RECORDS_PREPARATION.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(receipt,ensure_ascii=False),flush=True)
