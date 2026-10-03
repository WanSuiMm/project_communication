"""Export compact saved-case evidence; no training or model inference.

Per-map dictionary rows become column arrays without changing their values.
Large state/trace archives and machine launch metadata stay local.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
from joint195_analysis_stream import extract

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'evidence/joint195_200_20261003'
MANIFEST=ROOT/'JOINT195_PUBLICATION_MANIFEST.json'
RUN=ROOT/'runs/audit195_200_20261003_joint01'
METRICS={'strict128':('endpoints','128','strict'),
         'acquisition64_128':('intervals','64_128','primary','acquisition'),
         'first_exit64_128':('intervals','64_128','primary','first_exit'),
         'survival64_256':('intervals','64_256','primary','continuous_survival')}
OMITTED={'scope','population','observation_note','counting_note','schema_version'}
SEMANTICS={}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value,minified=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    options={'separators':(',',':')} if minified else {'indent':2}
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,**options)+'\n',encoding='utf-8')


def compact(value):
    if isinstance(value,dict):
        return {key:compact(v) for key,v in value.items() if key not in OMITTED}
    if isinstance(value,list):
        if value and all(isinstance(v,dict) and 'map_index' in v for v in value):
            rows=[{k:v for k,v in row.items() if k not in OMITTED} for row in value]
            if all(set(row)==set(rows[0]) for row in rows):
                keys=sorted(rows[0])
                return {'column_arrays':{key:[compact(row[key]) for row in rows] for key in keys}}
        return [compact(v) for v in value]
    return value


def stripped(value):
    """Independent canonicalization for a full numeric/value round trip."""
    if isinstance(value,dict):return {k:stripped(v)for k,v in value.items()if k not in OMITTED}
    if isinstance(value,list):return [stripped(v)for v in value]
    return value


def expanded(value):
    if isinstance(value,dict):
        if set(value)=={'column_arrays'}:
            columns=value['column_arrays'];lengths={len(v)for v in columns.values()}
            assert len(lengths)==1,'Unequal column-array lengths'
            return [{k:expanded(v[i])for k,v in columns.items()}for i in range(next(iter(lengths)))]
        return {k:expanded(v)for k,v in value.items()}
    if isinstance(value,list):return [expanded(v)for v in value]
    return value


def checked_compact(value):
    def notes(item):
        if isinstance(item,dict):
            for k,v in item.items():
                if k in OMITTED:
                    encoded=json.dumps(v,ensure_ascii=False,sort_keys=True)
                    SEMANTICS.setdefault(k,{})[encoded]=v
                notes(v)
        elif isinstance(item,list):
            for v in item:notes(v)
    notes(value)
    result=compact(value)
    assert expanded(result)==stripped(value),'Compact scientific-value round trip failed'
    return result


def metric(summary,path):
    if summary is None:return None
    for key in path:summary=summary[key]
    return summary['equal_map_mean']


def figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def case(name):return read(RUN/'cases'/f'{name}.json')
    def endpoint(name,t):return case(name)['summary']['endpoints'][str(t)]['strict']['equal_map_mean']
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    values=[endpoint(f'primary32_crossS{s}_D{d}_R195',256) for s,d in itertools.product((195,200),repeat=2)]
    axes[0].bar(range(4),values,color=['#777777','#777777','#28866c','#28866c'])
    axes[0].set_xticks(range(4),['S195\nD195','S195\nD200','S200\nD195','S200\nD200'])
    axes[0].set_ylim(.88,1.02);axes[0].set_title('Primary32: state origin / continuation')
    axes[0].set_ylabel('Strict T256 coverage, fixed R195 (zoomed)')
    for i,value in enumerate(values):axes[0].text(i,value+.002,f'{value:.4f}',ha='center',fontsize=9)
    for bank in ('primary32','primary64','confirmation32','confirmation64'):
        names=[f'{bank}_baseline195',*[f'{bank}_lambda{i:02d}'for i in range(1,10)],f'{bank}_baseline200']
        y=[case(n)['summary']['intervals']['64_128']['primary']['first_exit']['equal_map_mean']for n in names]
        axes[1].plot([i/10 for i in range(11)],y,marker='o',markersize=3,label=bank)
    axes[1].set_title('Interpolation: first-exit risk64--128')
    axes[1].set_xlabel('lambda195 -> 200');axes[1].set_ylabel('Equal-map risk')
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    old=[endpoint('primary32_factorial_none',128),endpoint('primary32_factorial_EF',128)]
    new=[endpoint('primary32_factorial_Q',128),endpoint('primary32_factorial_EFQ',128)]
    axes[2].bar([-.16,.84],old,width=.32,label='Q195')
    axes[2].bar([.16,1.16],new,width=.32,label='Q200')
    axes[2].set_xticks([0,1],['E/F195','E/F200']);axes[2].set_ylim(.85,1.02)
    axes[2].set_ylabel('Strict T128 coverage, fixed R195 (zoomed)')
    axes[2].set_title('Primary32: conditional Q effect')
    axes[2].legend(fontsize=9)
    for x,y in zip([-.16,.84,.16,1.16],[*old,*new]):axes[2].text(x,y+.003,f'{y:.4f}',ha='center',fontsize=8)
    fig.suptitle('One selected training trajectory; task-map confirmation is not training replication',fontsize=11)
    fig.tight_layout();fig.savefig(EVIDENCE/'mechanism.png',dpi=180);plt.close(fig)


def results(status):
    def case(name):return read(RUN/'cases'/f'{name}.json')
    def value(row,path):return metric(row['summary'],path)
    lines=['# Joint195/200: conditional state-production and coadaptation audit','',
           f"COMPLETE492/492, zero training, {status['elapsed_seconds']/60:.2f} minutes. Four full historical endpoint traces replay exactly; all counterfactuals are finite.",'',
           'Start with this report and [interpretation](INTERPRETATION.md). Complete numerical routing is [profiles.csv](profiles.csv), [case_index.json](case_index.json), and [factorial.json](factorial.json).',
           '', '## Endpoints', '',
           'All values below are equal-map percentages. Primary means changed/open/non-source (d>0); strict means16<d<32. Eligible maps and per-map counts remain in each compact case. Earlier dense reports also used pooled rates and included sources in turnover; compare matching estimands.', '',
           '| Bank | Checkpoint | Strict T128 | Acquisition64--128 | First-exit64--128 | Survival64--256 |',
           '|---|---:|---:|---:|---:|---:|']
    for bank in ('primary32','primary64','confirmation32','confirmation64'):
        for u in (195,200):
            row=case(f'{bank}_baseline{u}')
            v=[value(row,p)*100 for p in METRICS.values()]
            lines.append(f'| {bank} | {u} | '+ ' | '.join(f'{a:.4f}'for a in v)+' |')
    lines += ['', '## Cross-continuation', '', 'Primary32, fixed readout195. Hidden dynamics do not depend on the readout.', '',
              '| State at64 | Continuation | Strict T128 | Strict T256 | Shared-solved all-steps survival64--256 |',
              '|---:|---:|---:|---:|---:|']
    for s,d in itertools.product((195,200),repeat=2):
        row=case(f'primary32_crossS{s}_D{d}_R195');sm=row['summary']
        v=[sm['endpoints'][str(t)]['strict']['equal_map_mean']*100 for t in (128,256)]
        v += [sm['cohorts']['common_solved64']['all_steps_correct']['64_256']['equal_map_mean']*100]
        lines.append(f'| {s} | {d} | '+' | '.join(f'{a:.4f}'for a in v)+' |')
    lines += ['', 'The200-produced state can succeed under195 dynamics, while200 dynamics do not fully repair the195-produced state. This locates conditional pre64 state-production sensitivity; it does not make the continuation rule irrelevant.', '',
              '## Full factorial, not a standalone Q verdict', '',
              'Primary32, readout195. Other blocks stay195 unless listed.', '',
              '| Blocks from200 | Strict T64 | Strict T128 | Strict T256 | First-exit64--128 |',
              '|---|---:|---:|---:|---:|']
    for coalition in ('none','E','F','Q','EF','EFQ'):
        sm=case(f'primary32_factorial_{coalition}')['summary']
        v=[sm['endpoints'][str(t)]['strict']['equal_map_mean']*100 for t in (64,128,256)]
        v += [sm['intervals']['64_128']['primary']['first_exit']['equal_map_mean']*100]
        lines.append(f'| {coalition} | '+' | '.join(f'{a:.4f}'for a in v)+' |')
    lines += ['', 'Q200 has a different effect in matching E/F200 versus E/F195 backgrounds. [All16 coalitions and24 orders](factorial.json) are retained, including large negative mixtures.', '',
              '![Canonical contrast plots](mechanism.png)', '',
              '## Limits and validation', '',
              'The primary32 interpolation improvement at.4->.5 mainly removes map10 failure.195 already performs well on independent32 maps. These banks replicate task-map checks, not training trajectories. Sparse local interventions and nonzero solved-state updates do not identify a unique local commitment law, physical phase transition or universally equivalent representations.', '',
              '[Saved-artifact validation](validation/local_validation.json) uses CPU only. Checkpoints, states, raw NPZ traces, private execution receipts and the large duplicate analysis stay local; [provenance](input_provenance.json) retains their bindings. [Reproduction](REPRODUCTION.md) distinguishes compact public arithmetic checks from GPU inference. Earlier architecture no-go verdicts are unchanged.']
    (EVIDENCE/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def export(validation):
    assert not EVIDENCE.exists(),'New evidence directory required'
    status,aggregate,meta=read(RUN/'status.json'),read(RUN/'aggregate.json'),read(RUN/'manifest.json')
    checked=read(validation)
    assert status['status']=='COMPLETE' and aggregate['pass'] and aggregate['cases']==492
    assert checked['validation_status']=='PASS','Saved-result validation must pass before publication'
    EVIDENCE.mkdir(parents=True)
    for folder in ('cases','updates','plans','validation'):(EVIDENCE/folder).mkdir()
    slices=ROOT/'analyses/joint195_200_analysis_slices01.json'
    if slices.exists():
        analysis=read(slices)
        assert analysis['input_analysis_sha256']==sha(RUN/'analysis.json'),'Analysis cache input drift'
    else:
        analysis=extract(RUN/'analysis.json')
    effects=analysis['matched_baselines_by_case']
    index=[]
    for path in sorted((RUN/'cases').glob('*.json')):
        row=read(path)
        allowed=('case','group','bank','size','confirmation','steps','zero_training','extra','scientific_outcome','nonfinite_at_or_before')
        public={k:row[k] for k in allowed if k in row}
        public['summary']=checked_compact(row.get('summary'))
        public['original_case_sha256']=sha(path)
        public['original_trace_sha256']=row.get('trace_sha256',row.get('failed_trace_sha256'))
        public['bank_sha256']=row.get('bank_sha256')
        if row['case'] in effects:
            effect=effects[row['case']]
            public['matched_baseline_case']=effect['baseline_case']
            public['matched_baseline_cohorts']=checked_compact({name:item['baseline']for name,item in effect['cohort_effects'].items()})
        write(EVIDENCE/'cases'/path.name,public,minified=True)
        index.append({k:public.get(k) for k in ('case','group','bank','size','confirmation')}|
                     {'metrics_equal_map':{m:metric(row.get('summary'),p)for m,p in METRICS.items()},
                      'file':f'cases/{path.name}','status':row.get('scientific_outcome','FINITE')})
    assert len(index)==492
    for sub in ('updates','plans'):
        for path in sorted((RUN/sub).glob('*.json')):write(EVIDENCE/sub/path.name,checked_compact(read(path)),minified=True)
    write(EVIDENCE/'schema_semantics.json',{
        'centralized_fields':{k:list(v.values())for k,v in sorted(SEMANTICS.items())},
        'copy_rule':'All retained scientific values round-trip exactly. Repeated metadata prose is centralized here; per-map rows use lossless column arrays.',
        'roundtrip_checked':True,'cases':492,'updates':96,'plans':16})
    write(EVIDENCE/'case_index.json',{'schema':'joint195_compact_index_v1','cases':index,
        'reporting_unit':'map; one selected training trajectory','column_encoding':'per_map dictionary rows encoded as column_arrays; values unchanged'})
    # Factorial order effects are small; the huge duplicate localized-effect tree is excluded.
    write(EVIDENCE/'factorial.json',analysis['factorial_by_bank'])
    write(EVIDENCE/'confirmation_agreement.json',analysis['confirmation_agreement'])
    with (EVIDENCE/'profiles.csv').open('w',encoding='utf-8',newline='') as f:
        fields=['case','group','bank','size','confirmation',*METRICS]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for item in index:writer.writerow({k:item[k]for k in fields if k not in METRICS}|item['metrics_equal_map'])
    # Expose only scientific/runtime settings, not actual host/PID/command data.
    write(EVIDENCE/'run_metadata.json',{'protocol':meta['protocol'],'execution':'COMPLETE','cases':492,
          'execution_seconds':status['elapsed_seconds'],'training_trajectories':1,'new_training':False,
          'torch':meta['torch'],'threads':meta['threads'],'backend':meta['backend'],
          'parameter_sha256':meta['bindings']['parameter_sha256'],
          'source_sha256':meta['bindings']['source_sha256'],'audit_banks':meta['audit_banks'],
          'replay_controls':aggregate['replay_controls'],'maximum_seconds':None,'runtime_limit_enforced':False})
    # Validator's schema is intentionally CPU scientific evidence only.
    public_validation={k:v for k,v in checked.items() if k not in ('run','out','host','pid','argv','command','stdout','stderr')}
    write(EVIDENCE/'validation/local_validation.json',public_validation)
    shutil.copyfile(RUN/'analysis.png',EVIDENCE/'analysis.png')
    figure()
    (EVIDENCE/'RECORDED_RESULTS.md').write_text(
        'Publication note: the original generated report follows verbatim. Its `analysis.json`\n'
        'reference denotes the excluded large local analysis tree; public replacements are\n'
        '`factorial.json`, `case_index.json` and the compact `cases/` records. The derived\n'
        '`DATA_COMPLETE` label predates the final `COMPLETE` status write. Start at `RESULTS.md`.\n\n'
        +(RUN/'RESULTS.md').read_text(encoding='utf-8'),encoding='utf-8')
    original={f'cases/{p.name}':sha(p)for p in sorted((RUN/'cases').glob('*.json'))}
    original.update({name:sha(RUN/name)for name in ('status.json','aggregate.json','summary.json','analysis.json')})
    write(EVIDENCE/'input_provenance.json',{'original_record_sha256':original,
        'original_input_sha256':meta['bindings']['input_sha256'],
        'excluded':'checkpoints, all state/trace NPZ archives, 180MB summary, 420MB duplicate analysis tree, source snapshots, private launch receipts and logs',
        'compact_copy_rule':'retained scientific values round-trip exactly; repeated prose centralized in schema_semantics.json and per-map rows column encoded; original byte hashes retained'})
    write(EVIDENCE/'summary.json',{'execution':'COMPLETE','cases':492,'execution_seconds':status['elapsed_seconds'],
        'new_training':False,'training_trajectories':1,'case_counts':analysis['case_count_check'],
        'all_arms_finite':all(item['status']=='FINITE' for item in index),
        'case_index':'case_index.json','profiles':'profiles.csv','factorial':'factorial.json',
        'reading_order':['RESULTS.md','INTERPRETATION.md','profiles.csv','case_index.json'],
        'claim':'conditional state-production and coadaptation evidence; no universal mechanism or physical phase transition'})
    results(status)
    print(json.dumps({'exported_cases':len(index),'evidence':EVIDENCE.relative_to(ROOT).as_posix()}))


def finalize():
    meta=read(RUN/'manifest.json')
    files=sorted(p for p in EVIDENCE.rglob('*')if p.is_file())
    source=meta['bindings']['source_sha256']
    for name,expected in source.items():assert sha(ROOT/name)==expected,f'Executed source drift:{name}'
    tools=['new/audit_195_200/validate_saved.py','tools/export_joint195_evidence.py',
           'tools/joint195_analysis_stream.py','tools/replay_joint195_public.py']
    write(MANIFEST,{'schema':'joint195_publication_v1','review_base':'d52bf4729605f9adf55c8579c47a4cf593c78a4f',
        'protocol':'audit195_200_v1','source_sha256':source,
        'publication_tool_sha256':{name:sha(ROOT/name)for name in tools},
        'published_evidence_sha256':{p.relative_to(ROOT).as_posix():sha(p)for p in files},
        'excluded_artifacts':'original run archives and machine receipts remain local',
        'scientific_scope':'one selected checkpoint pair;48 diagnostic maps over two sizes; zero new training'})


def verify():
    manifest=read(MANIFEST)
    for key in ('source_sha256','publication_tool_sha256','published_evidence_sha256'):
        for name,expected in manifest[key].items():assert sha(ROOT/name)==expected,f'Hash drift:{name}'
    index=read(EVIDENCE/'case_index.json')['cases']
    assert len(index)==492 and len({r['case']for r in index})==492
    rate_nodes=0
    def rates(value):
        nonlocal rate_nodes
        if isinstance(value,dict):
            if 'pooled' in value and 'equal_map_mean' in value and 'per_map' in value:
                per=value['per_map']['column_arrays'];num=per['numerator'];den=per['denominator']
                assert sum(num)==value['pooled']['numerator'] and sum(den)==value['pooled']['denominator']
                observed=[n/d for n,d in zip(num,den)if d]
                expect=sum(observed)/len(observed)if observed else None
                actual=value['equal_map_mean']
                assert actual is None and expect is None or actual is not None and abs(actual-expect)<2e-7
                rate_nodes+=1
            for item in value.values():rates(item)
        elif isinstance(value,list):
            for item in value:rates(item)
    for item in index:
        row=read(EVIDENCE/item['file']);assert row['case']==item['case']
        assert {m:metric(row['summary'],p)for m,p in METRICS.items()}==item['metrics_equal_map']
        rates(row['summary'])
        rates(row.get('matched_baseline_cohorts'))
    factorial=read(EVIDENCE/'factorial.json')
    for bank,values in factorial.items():
        for name,result in values['shapley'].items():
            assert result['complete'] and len(result['permutation_orders'])==24
            means=[result['by_block'][b]['mean']for b in 'EFQR']
            endpoints=values['values_by_metric'][name]
            assert abs(sum(means)-(endpoints['EFQR']-endpoints['none']))<2e-7
    print(json.dumps({'status':'PASS','cases':492,'rate_nodes':rate_nodes,'factorial_banks':len(factorial)}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation',default='analyses/joint195_200_20261003_validation03.json')
    p.add_argument('--finalize',action='store_true');p.add_argument('--verify-only',action='store_true')
    args=p.parse_args()
    if args.verify_only:verify()
    elif args.finalize:finalize();verify()
    else:export(ROOT/args.validation)


if __name__=='__main__':main()
