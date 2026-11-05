#!/usr/bin/env python3
"""Recompute the numbers reported in the article from the five source CSV files.

Standard library only (Python 3.10+). Does not use the network and does not change the input.

    python audit_repro.py --data parking_team_package/data \
        --refs parking_team_package/tables/reference_answers.csv --out audit_results.json
"""
import argparse, csv, json, math, statistics
from collections import defaultdict
from datetime import date
from pathlib import Path

MONTHS = ['202605', '202606', '202607', '202608', '202609']
NAMES = dict(zip(MONTHS, ['May', 'June', 'July', 'August', 'September']))


def load(data_dir):
    rows = []
    for m in MONTHS:
        with open(Path(data_dir) / f'parking_places_hourly_{m}.csv', encoding='utf-8', newline='') as fh:
            for r in csv.DictReader(line for line in fh if line.strip()):
                rows.append(dict(month=m, day=r['dateServ'], hour=int(r['hour']), id=int(r['idParking']),
                                 emt=r['isEmtPark'] == 'True', f=int(r['freeParking']), c=int(r['parkingSpaces'])))
    return rows


def valid(r):
    return 0 <= r['f'] <= r['c'] and r['c'] > 0


def q_linear(x, p):
    h = (len(x) - 1) * p
    a, b = math.floor(h), math.ceil(h)
    return x[a] + (h - a) * (x[b] - x[a])


def q_exclusive(x, p):  # Hyndman-Fan type 6, h = (n + 1) p
    h = min(max((len(x) + 1) * p, 1), len(x)) - 1
    a, b = math.floor(h), math.ceil(h)
    return x[a] + (h - a) * (x[b] - x[a])


def q_nearest(x, p):  # nearest rank, Hyndman-Fan type 1
    return x[max(math.ceil(len(x) * p) - 1, 0)]


QUANT = dict(linear=q_linear, exclusive=q_exclusive, nearest=q_nearest)


def frozen_runs(p1, L):
    """Maximal runs of identical free counts per car park over valid rows; invalid rows do not break a run."""
    by = defaultdict(list)
    for r in p1:
        by[r['id']].append(r)
    runs = []
    for pid, obs in by.items():
        obs.sort(key=lambda r: (r['day'], r['hour']))
        i = 0
        while i < len(obs):
            j = i
            while j + 1 < len(obs) and obs[j + 1]['f'] == obs[i]['f']:
                j += 1
            if j - i + 1 >= L:
                runs.append(dict(id=pid, value=obs[i]['f'], cap=obs[i]['c'], n=j - i + 1,
                                 start=obs[i]['day'], end=obs[j]['day'], rows=obs[i:j + 1]))
            i = j + 1
    return runs


def occ_rows(rs):
    return math.fsum(1 - r['f'] / r['c'] for r in rs) / len(rs) if rs else None


def occ_w(rs):
    return 1 - sum(r['f'] for r in rs) / sum(r['c'] for r in rs) if rs else None


def group(rs, key):
    d = defaultdict(list)
    for r in rs:
        d[key(r)].append(r)
    return d


def tops(rs):
    g = {k: occ_rows(v) for k, v in group(rs, lambda r: r['id']).items()}
    best = max(g.values())
    return sorted(k for k, v in g.items() if abs(v - best) < 1e-12), best


def hour_rank(rs):
    g = sorted(((occ_rows(v), k) for k, v in group(rs, lambda r: r['hour']).items()), reverse=True)
    return g


def iqr_count(rs, q, by_car_park=True):
    n = 0
    groups = group(rs, lambda r: r['id']).values() if by_car_park else [rs]
    for g in groups:
        x = sorted(1 - r['f'] / r['c'] for r in g)
        a, b = q(x, .25), q(x, .75)
        lo, hi = a - 1.5 * (b - a), b + 1.5 * (b - a)
        n += sum(v < lo or v > hi for v in x)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--refs', required=True)
    ap.add_argument('--out', default='audit_results.json')
    a = ap.parse_args()
    rows = load(a.data)
    p1 = [r for r in rows if valid(r)]
    R = {}
    # ---- defects
    cap = {r['id']: r['c'] for r in rows}
    working = sorted({r['id'] for r in p1})
    over = [r for r in rows if r['f'] > r['c']]
    R['defects'] = dict(rows=len(rows), minus1=sum(r['f'] == -1 for r in rows), below_minus1=sum(r['f'] < -1 for r in rows),
                        below_minus1_ids=sorted({r['id'] for r in rows if r['f'] < -1}),
                        below_range=[min(r['f'] for r in rows), max(r['f'] for r in rows if r['f'] < -1)],
                        over=len(over), over_id32=sum(r['id'] == 32 for r in over), p1=len(p1),
                        parks=len(cap), working=len(working), cap_total=sum(cap.values()),
                        cap_working=sum(cap[i] for i in working), emt_true=sum(1 for i in cap if {r['emt'] for r in rows if r['id'] == i} == {True}))
    # ---- frozen
    runs24, runs12 = frozen_runs(p1, 24), frozen_runs(p1, 12)
    per = defaultdict(list)
    for r in runs24:
        per[r['id']].append((r['n'], r['value'], r['start'], r['end']))
    R['frozen'] = dict(runs_L24=len(runs24), rows_L24=sum(r['n'] for r in runs24), rows_L12=sum(r['n'] for r in runs12),
                       runs_L12=len(runs12), by_id={str(k): sorted(v, reverse=True) for k, v in sorted(per.items())})
    flag = {(r['id'], x['day'], x['hour']) for r in runs24 for x in r['rows']}
    flag12 = {(r['id'], x['day'], x['hour']) for r in runs12 for x in r['rows']}
    ps = [r for r in p1 if (r['id'], r['day'], r['hour']) not in flag]
    ps12 = [r for r in p1 if (r['id'], r['day'], r['hour']) not in flag12]
    id80 = [r['f'] for r in p1 if r['id'] == 80]
    R['id80'] = dict(n=len(id80), zero=id80.count(0), nonzero=len(id80) - id80.count(0), min_nz=min(v for v in id80 if v), max=max(id80),
                     occ_at=[round(1 - v / 3027, 3) for v in (1611, 1719)], share_working=cap[80] / R['defects']['cap_working'])
    # ---- monthly
    M = {}
    for m in MONTHS:
        x = [r for r in p1 if r['month'] == m]
        y = [r for r in ps if r['month'] == m]
        z = [r for r in ps12 if r['month'] == m]
        no80 = [r for r in x if r['id'] != 80]
        hx, hy = hour_rank(x), hour_rank(y)
        tx, ty = tops(x), tops(y)
        M[NAMES[m]] = dict(p1=len(x), excluded=len(x) - len(y), excluded_L12=len(x) - len(z), mean_free=statistics.fmean(r['f'] for r in x),
                           occ_rows=occ_rows(x), occ_w=occ_w(x), s_rows=occ_rows(y), s_w=occ_w(y), s12_w=occ_w(z),
                           w_without_80=occ_w(no80), rows_without_80=occ_rows(no80), peak=hx[0][1], peak_v=hx[0][0], second=hx[1][1], second_v=hx[1][0],
                           s_peak=hy[0][1], top=tx[0], top_v=tx[1], s_top=ty[0], s_top_v=ty[1])
    R['monthly'] = M
    # ---- sensitivity
    with open(a.refs, encoding='utf-8-sig', newline='') as fh:
        refs = list(csv.DictReader(fh))
    frac = [r for r in refs if float(r['absolute_tolerance']) > 0]
    S = {'fractional': len(frac)}
    def is_num(v):
        try:
            float(v); return True
        except ValueError:
            return False
    for d in (2, 3, 4, 6):
        S[f'broken_round_{d}'] = sum(abs(round(float(r['expected_value']), d) - float(r['expected_value'])) > float(r['absolute_tolerance']) for r in frac if is_num(r['expected_value']))
    shares = [r for r in frac if r['unit'] == 'fraction_0_1']
    S['shares'] = len(shares)
    S['broken_pct2'] = sum(abs(round(float(r['expected_value']) * 100, 2) / 100 - float(r['expected_value'])) > float(r['absolute_tolerance']) for r in shares)
    S['max_err_round4'] = max(abs(round(float(r['expected_value']), 4) - float(r['expected_value'])) for r in frac)
    S['iqr_question'] = next(r['question_ru'] for r in refs if r['metric'] == 'iqr_flag_n')
    S['free_q1_question'] = next(r['question_ru'] for r in refs if r['metric'] == 'free_q1')
    S['free_std_question'] = next(r['question_ru'] for r in refs if r['metric'] == 'free_std_sample')
    D = {}
    for m in MONTHS:
        x = [r for r in p1 if r['month'] == m]
        f = [r['f'] for r in x]
        s1, s0 = statistics.stdev(f), statistics.pstdev(f)
        srt = sorted(f)
        ge0 = [r for r in rows if r['month'] == m and r['f'] >= 0 and r['c'] > 0]
        raw = [r for r in rows if r['month'] == m]
        D[NAMES[m]] = dict(sd1=s1, sd_diff=s1 - s0, rel=(s1 - s0) / s1,
                           q={k: [q(srt, .25), q(srt, .75)] for k, q in QUANT.items()},
                           iqr={k: iqr_count(x, q) for k, q in QUANT.items()}, iqr_ungrouped=iqr_count(x, q_linear, False),
                           median=statistics.median(f), ge0_median=statistics.median(r['f'] for r in ge0), ge0_occ=occ_rows(ge0), occ=occ_rows(x),
                           raw_occ=occ_rows(raw), raw_median=statistics.median(r['f'] for r in raw))
    S['by_month'] = D
    R['sensitivity'] = S
    # ---- missing values
    work_emt = [i for i in working if any(r['emt'] for r in rows if r['id'] == i)]
    work_non = [i for i in working if i not in work_emt]
    em = [r for r in rows if r['id'] in work_emt]
    allmiss = defaultdict(int)
    for r in em:
        allmiss[(r['month'], r['day'], r['hour'])] += r['f'] == -1
    sysw = [k for k, v in allmiss.items() if v == len(work_emt)]
    part = [k for k, v in allmiss.items() if 0 < v < len(work_emt)]
    ph = defaultdict(int)
    for k in sysw:
        ph[k[2]] += 1
    bymonth = defaultdict(int)
    for k in sysw:
        bymonth[NAMES[k[0]]] += 1
    non = [r for r in rows if r['id'] in work_non]
    hm = {NAMES[m]: abs(statistics.fmean(v for _, v in [(0, occ_rows(g)) for g in group([r for r in p1 if r['month'] == m], lambda r: r['hour']).values()]) - occ_rows([r for r in p1 if r['month'] == m])) for m in MONTHS}
    R['missing'] = dict(work_emt=len(work_emt), work_non=len(work_non), emt_minus1=sum(r['f'] == -1 for r in em) / len(em), date_hours=len(allmiss), system_wide=len(sysw), partial=len(part),
                        non_minus1=sum(r['f'] == -1 for r in non) / len(non), by_month=dict(bymonth), by_hour={k: ph[k] for k in range(24)}, hourly_vs_rows=hm)
    # ---- ID 32
    i32 = [r for r in rows if r['id'] == 32]
    o32 = [r for r in i32 if r['f'] > r['c']]
    bm = defaultdict(int)
    for r in o32:
        bm[NAMES[r['month']]] += 1
    bh = defaultdict(int)
    for r in o32:
        bh[r['hour']] += 1
    nn = [r for r in i32 if r['f'] >= 0]
    clip = [1 - min(r['f'], r['c']) / r['c'] for r in nn]
    R['id32'] = dict(cap=i32[0]['c'], max=max(r['f'] for r in i32), over=len(o32), by_month=dict(bm), by_hour={k: bh[k] for k in sorted(bh)}, share_nonneg=len(o32) / len(nn), occ_p1=occ_rows([r for r in i32 if valid(r)]), occ_clip=statistics.fmean(clip))
    # ---- coverage
    stable = [i for i in working if i not in (15, 49, 80)]
    C = {}
    for m in MONTHS:
        x = [r for r in p1 if r['month'] == m]
        st = [r for r in x if r['id'] in stable]
        eq = statistics.fmean(occ_rows(v) for v in group(st, lambda r: r['id']).values())
        cnt = group(st, lambda r: (r['day'], r['hour']))
        bal = [r for v in cnt.values() if len(v) == len(stable) for r in v]
        cell = group(x, lambda r: (date.fromisoformat(r['day']).weekday(), r['hour']))
        std = statistics.fmean(occ_rows(v) for v in cell.values())
        cell27 = group(st, lambda r: (date.fromisoformat(r['day']).weekday(), r['hour']))
        std27 = statistics.fmean(occ_rows(v) for v in cell27.values())
        wk27 = [r for r in st if date.fromisoformat(r['day']).weekday() < 5]
        sun27 = [r for r in st if date.fromisoformat(r['day']).weekday() == 6]
        wk = [r for r in x if date.fromisoformat(r['day']).weekday() < 5]
        sun = [r for r in x if date.fromisoformat(r['day']).weekday() == 6]
        C[NAMES[m]] = dict(p1=occ_rows(x), stable=eq, balanced=occ_rows(bal), standardised_all=std, standardised=std27, weekday_minus_sunday_27=occ_rows(wk27) - occ_rows(sun27), p1s=occ_rows([r for r in ps if r['month'] == m]), weekday_minus_sunday=occ_rows(wk) - occ_rows(sun), stable_datehours=sum(1 for v in cnt.values() if len(v) == len(stable)))
    R['coverage'] = C
    below, minimum = 0, 0
    for i in stable:
        o = {NAMES[m]: occ_rows([r for r in p1 if r['month'] == m and r['id'] == i]) for m in MONTHS}
        rest = statistics.fmean(v for k, v in o.items() if k != 'August')
        below += o['August'] < rest
        minimum += o['August'] == min(o.values())
    n = len(stable)
    R['august'] = dict(stable=n, below=below, minimum=minimum, sign_test_two_sided=2 * sum(math.comb(n, k) for k in range(below, n + 1)) / 2 ** n)
    Path(a.out).write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print('written', a.out)


if __name__ == '__main__':
    main()
