"""Recompute a post-run loss audit from original logs and saved predictions only."""
import csv
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
KIT=ROOT.parent/'rebot-b601-training/openpi_b601'
MON=KIT/'runs/pi05_b601_lora/b601-final60-v1_monitor'
OUT=ROOT/'LOSS_AUDIT_20261110'
OUT.mkdir(exist_ok=False)


def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_csv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


train=lines(ROOT/'CONSOLE_TRAIN_METRICS.jsonl')
val=lines(MON/'VALIDATION_METRICS.jsonl')
assert [r['step_label'] for r in train]==list(range(0,43001,100))
assert [r['step_label'] for r in val]==list(range(1000,43001,1000))
raw={}
for match in re.finditer(r'Step\s+(\d+):\s*([^\r\n]+)',(ROOT/'logs/formal-longrun-v1.log').read_text()):
    fields=dict(re.findall(r'(loss|grad_norm|param_norm)=([^,\s]+)',match.group(2)))
    if len(fields)==3:raw[int(match.group(1))]={k:float(v) for k,v in fields.items()}
assert len(raw)==431
assert all(all(raw[r['step_label']][k]==r[k] for k in ('loss','grad_norm','param_norm')) for r in train)
assert all(np.isfinite([r[k] for r in train for k in ('loss','grad_norm','param_norm')]))
changes=np.diff([r['loss'] for r in train])
minimum=min(r['loss'] for r in train)
phases=[]
for lo,hi in [(0,1000),(1000,5000),(5000,10000),(10000,20000),(20000,30000),(30000,40000),(40000,43000)]:
    values=[r['loss'] for r in train if lo<r['step_label']<=hi]
    phases.append({'after_step':lo,'through_step':hi,'logged100update_intervals':len(values),
                   'mean_of_rounded_interval_means':float(np.mean(values)),
                   'min_recorded_interval_mean':min(values),'max_recorded_interval_mean':max(values)})
protocol=read(MON/'EVAL_PROTOCOL.json')
episodes=lines(KIT/'prepared/datasets/rebot-b601-val/meta/episodes.jsonl')
lengths={r['episode_index']:r['length'] for r in episodes}
flow_total=len(protocol['observations'])*25
flow_valid=sum(min(25,lengths[r['local_episode']]-r['frame']) for r in protocol['observations'])
assert (flow_total,flow_total-flow_valid)==(2000,240)

metrics=[];per_episode=[];source_predictions={};canonical=None
for step,name in [(5000,'best'),(10000,'checkpoint10000'),(43000,'last')]:
    path=ROOT/'evaluation'/name/'predictions.npz';z=np.load(path)
    prediction=z['prediction'].astype(np.float64);target=z['target'].astype(np.float64)
    mask=z['valid_mask'];ids=z['episode_source_frame']
    if canonical is None:canonical=(target.copy(),mask.copy(),ids.copy())
    else:
        for actual,expected in zip((target,mask,ids),canonical):np.testing.assert_array_equal(actual,expected)
    assert prediction.shape==(211,25,7) and int(mask.sum())==5118 and np.isfinite(prediction).all()
    errors=(prediction-target)[mask];axis=np.abs(errors).mean(0)
    t=next(r['loss'] for r in train if r['step_label']==step)
    v=next(r for r in val if r['step_label']==step)
    metrics.append({'step_label':step,'optimizer_updates':step+1,'active_hours':v['active_seconds']/3600,
                    'training_recorded100step_mean':t,'validation_flow_loss':v['validation_flow_loss'],
                    'joint6_mean_mae_float64':float(axis[:6].mean()),'gripper_mae_float64':float(axis[6])})
    for episode in sorted(np.unique(ids[:,0])):
        rows=ids[:,0]==episode;error=(prediction[rows]-target[rows])[mask[rows]]
        per_episode.append({'step_label':step,'source_episode':int(ids[rows][0,1]),'valid_action_rows':len(error),
                            'joint6_mean_mae':float(np.abs(error[:,:6]).mean()),'gripper_mae':float(np.abs(error[:,6]).mean())})
    source_predictions[str(step)]=sha(path)
early,last=metrics[1],metrics[2]
per10={r['source_episode']:r for r in per_episode if r['step_label']==10000}
per43={r['source_episode']:r for r in per_episode if r['step_label']==43000}
gains={key:100*(1-last[key]/early[key]) for key in ('joint6_mean_mae_float64','gripper_mae_float64')}
counts={key:sum(per43[e][key]<per10[e][key] for e in per10) for key in ('joint6_mean_mae','gripper_mae')}
stats={'training_log_records':len(train),'validation_records':len(val),
    'raw_stdout_reparsed_exactly_matches_collector':True,
    'training_log_precision':'4 decimals; step0 single update, subsequent records100-update means; not every-step loss',
    'min_recorded_training_interval_mean':minimum,'min_recorded_at_steps':[r['step_label'] for r in train if r['loss']==minimum],
    'last_recorded_training_interval_mean':train[-1]['loss'],
    'adjacent_record_changes':{'up':int((changes>0).sum()),'down':int((changes<0).sum()),'equal_at4dp':int((changes==0).sum())},
    'phase_training_means':phases,'candidate_metrics':metrics,
    '10k_to43k_improvement_percent':gains,'10k_to43k_episodes_improved_of10':counts,
    'extra_active_hours_10k_to43k':last['active_hours']-early['active_hours'],
    'validation_flow_min':min(val,key=lambda r:r['validation_flow_loss']),
    'validation_flow_5k_to43k_increase_percent':100*(metrics[2]['validation_flow_loss']/metrics[0]['validation_flow_loss']-1),
    'flow_proxy_scope':{'observations':80,'dimension_count':32,'real_dimensions':7,'padded_dimensions':25,
                        'window_rows_including_terminal_repetition':flow_total,'artificial_terminal_repeated_rows':flow_total-flow_valid,
                        'terminal_padding_fraction':(flow_total-flow_valid)/flow_total,'rng_fixed_across_checkpoints':True},
    'action_metric_scope':{'chunks':211,'valid_rows':5118,'padded_rows_excluded':211*25-5118,
                          'dimensions':7,'sampling_steps':10,'distinct_trained_checkpoints':3,
                          'validation_reused_for_selection':True,'physical_grasp_success_measured':False},
    'source_sha256':{'console_metrics':sha(ROOT/'CONSOLE_TRAIN_METRICS.jsonl'),'formal_stdout':sha(ROOT/'logs/formal-longrun-v1.log'),
                     'validation_metrics':sha(MON/'VALIDATION_METRICS.jsonl'),'evaluation_predictions':source_predictions}}
(OUT/'LOSS_AUDIT_SUMMARY.json').write_text(json.dumps(stats,indent=2)+'\n')
write_csv('training_curve.csv',[{k:r[k] for k in ('step_label','loss','grad_norm','param_norm')} for r in train])
write_csv('validation_flow_curve.csv',[{k:r[k] for k in ('step_label','optimizer_step','validation_flow_loss','active_seconds')} for r in val])
write_csv('phase_training_means.csv',phases);write_csv('checkpoint_comparison.csv',metrics);write_csv('per_episode_comparison.csv',per_episode)

font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if Path(font_path).exists():
    fm.fontManager.addfont(font_path);plt.rcParams['font.family']=fm.FontProperties(fname=font_path).get_name()
plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'pdf.fonttype':42,'svg.fonttype':'path'})
fig,axes=plt.subplots(2,2,figsize=(14,8.3),constrained_layout=True)
x=np.array([r['step_label'] for r in train]);y=np.array([r['loss'] for r in train])
ax=axes[0,0];ax.plot(x,y,color='#74a9cf',lw=.8,alpha=.65,label='原始100步区间均值（0号为单步）')
ax.plot(x[10:],np.convolve(y[1:],np.ones(10)/10,'valid'),color='#08519c',lw=2,label='约1000步滑动均值')
ax.set_yscale('log');ax.set_title('训练 loss：总体下降，后期接近平台');ax.set_ylabel('归一化32维 flow MSE（对数轴）')
ax.annotate('最低记录 0.0011（4位小数）',xy=(25500,.0011),xytext=(16000,.007),arrowprops={'arrowstyle':'->'},fontsize=9)
ax.legend(fontsize=8);ax.grid(alpha=.18)
ax=axes[0,1];vx=[r['step_label'] for r in val];vy=[r['validation_flow_loss'] for r in val]
ax.plot(vx,vy,'o-',color='#d95f0e',ms=3,lw=1.5);ax.set_title('固定80观察的验证 flow：5k后回升')
ax.set_ylabel('归一化32维 flow MSE');ax.set_ylim(.0053,.0105);ax.grid(alpha=.18)
ax.annotate('5k：0.005922',xy=(5000,vy[4]),xytext=(9000,.00545),arrowprops={'arrowstyle':'->'},fontsize=9)
ax.annotate('43k：0.009592',xy=(43000,vy[-1]),xytext=(26000,.01015),arrowprops={'arrowstyle':'->'},fontsize=9)
for ax,key,title,color,lim in [(axes[1,0],'joint6_mean_mae_float64','实际六关节平均 MAE','#238b45',(0,2.7)),
                               (axes[1,1],'gripper_mae_float64','实际夹爪 MAE','#756bb1',(0,2.4))]:
    values=[r[key] for r in metrics];ax.plot([5000,10000,43000],values,'o--',color=color,lw=1.5,ms=6)
    for point,value in zip((5000,10000,43000),values):ax.annotate(f'{value:.4f}',(point,value),xytext=(0,9),textcoords='offset points',ha='center')
    ax.set_title(title+f'：10k→43k下降 {gains[key]:.2f}%')
    ax.set_ylabel('原始动作坐标单位');ax.set_ylim(*lim);ax.set_xticks([5000,10000,43000],['5k','10k','43k']);ax.grid(alpha=.18)
for ax in axes.flat:ax.set_xlabel('训练保存点编号（优化器更新次数为编号+1）')
fig.suptitle('B601完整训练复核：训练loss下降，不等于所有验证指标持续改善\n下方虚线仅连接三个已评估候选，不代表其间所有模型的走势',fontsize=13)
for suffix in ('png','svg','pdf'):fig.savefig(OUT/f'loss_and_action_curves.{suffix}',dpi=180)
plt.close(fig)

table='\n'.join(f"| {r['step_label']:,} | {r['training_recorded100step_mean']:.4f} | {r['validation_flow_loss']:.6f} | {r['joint6_mean_mae_float64']:.6f} | {r['gripper_mae_float64']:.6f} |" for r in metrics)
report=f'''# Loss与训练流程复核（2026-11-10）

结论：训练loss总体下降，后期波动并趋于平台；固定验证flow在5k后回升。后续长跑对已比较的真实动作指标有收益，但不能说所有指标全程单调改善，更不能据此无限增加步数。

## 真实曲线与日志口径

![训练、验证与动作误差](loss_and_action_curves.png)

正式训练有431条记录：step0为单步，此后每100步输出区间均值，保留四位小数。已从原始formal-longrun-v1.log重新解析，loss/grad_norm/param_norm与采集JSONL逐条完全一致。不能从这些区间均值恢复43,001条单步全精度loss。

早期400、600、700步分别为0.0168、0.0125、0.0115，800步已到0.0094。最低**记录区间均值**为0.0011，出现在25,500、29,400、32,900、37,800、41,100；末条为0.0015。这不等于已知最低单步loss。430次相邻记录比较有179次上升、206次下降、45次在四位小数下相等。

训练按step重抽flow噪声/插值时刻、采样数据并做图像增强，batch只有4，逐条不单调正常。Dropout默认0，不把本例波动泛称为dropout造成。固定验证使用同80个观察、相同种子及批次键、关闭训练增强；5k后持续回升不能归因为每次重抽了另一批验证噪声。

## 对比真实动作误差

| 保存点 | 训练100步均值 | 验证flow | 六关节平均MAE | 夹爪MAE |
| ---: | ---: | ---: | ---: | ---: |
{table}

10k→43k额外运行{stats['extra_active_hours_10k_to43k']:.2f}活动小时，关节MAE下降{gains['joint6_mean_mae_float64']:.2f}%，10条验证episode均改善；夹爪MAE仅下降{gains['gripper_mae_float64']:.2f}%，5条改善、5条变差。关节收益较一致，夹爪收益小且不一致。5k→43k关节/夹爪分别改善18.74%/13.53%，虽然验证flow反而上升{stats['validation_flow_5k_to43k_increase_percent']:.2f}%。

所有动作指标已用保存的预测和目标在float64下复算；三个模型目标、mask、源episode/frame逐元素相同，211个窗口覆盖5,118有效动作行，每行计一次，157个尾部padding行被排除。两种动作单位分开报告，不把六关节和夹爪混成一个分数。这里只比较5k/10k/43k三个不同候选，未证明每个中间checkpoint都越来越好，或43k在全程全局最优。

## 发现的验证代理局限

1. 上游flow loss对quantile归一化后的32维向量场误差取MSE；机械臂实际7维，另外25维补零也参与loss。维度权重并不等于其实际误差贡献，现有聚合日志不能拆出各部分贡献。
2. 固定80观察每episode取8点，其中每个episode的末帧窗口有24个重复尾步。共2,000行中240行（12%）是人工重复尾部，未从flow loss剔除。真实动作MAE已排除尾部padding，计算口径正确。
3. 验证flow每episode8观察等权；动作整体MAE按5,118有效帧加权，并另存episode宏平均。两者覆盖和权重不同。
4. flow loss衡量一个随机插值时刻的向量场预测，动作MAE衡量10步积分后、反归一化后的真实7维端点误差，两者没有单调等价关系。验证单个固定随机draw/80观察也不是完整期望风险。

因此当前flow可保留为沿用上游目标的训练监控，但不能独自决定部署模型。验证回升是该目标泛化不足的警讯，不能仅凭代理差异排除过拟合，也不能仅凭不单调认定训练代码错误。尚未做维度/尾部/时刻分解消融，不编造“是哪一项造成62%回升”的归因。

## 流程复核与建议

未发现本次动作映射、训练/验证划分、训练统计复用或MAE公式错误。训练和验证样本分开，归一化只来自训练集；恢复、参数/优化器有限值读回、确定性推理与模型包哈希验收见原始审计。原执行monitor保存在服务器source_versions/formal_v1_before_wandb_hook_fix，本归档的[执行版本](../source_snapshots/executed_history/formal_v1_before_wandb_hook_fix/longrun_monitor.py)供对照；W&B禁用重绑定导致单步hook失效，当前loss曲线由100步控制台记录恢复，此限制已保留，不能补造单步数据。

后续应同时记录真实7维/有效时间mask的辅助flow、补齐维度flow，并定期评估实际关节/夹爪MAE，以任务指标和现场成功率安排早停。30k后训练均值已约0.0016的平台，10k后的夹爪收益很小；不建议只凭训练loss继续堆步数。此次43k模型在已评估候选的两个平均动作指标上均优，交付选择有实际证据。

验证集参与选模，只有10条episode；没有多随机种子或独立测试集，未测真机抓放成功率。此复核仅使用既有日志和预测，不重训、不改写已发布权重。源文件SHA、完整曲线CSV、区间统计和逐episode比较均在本目录。
'''
(OUT/'LOSS_AUDIT_CN.md').write_text(report)
print(json.dumps({'output':str(OUT),'raw_log_exact_match':True,'minimum':minimum,'metrics':metrics,
                  'episode_improvement_counts':counts,'phase_means':phases,'tail_padding_rows':240}))
