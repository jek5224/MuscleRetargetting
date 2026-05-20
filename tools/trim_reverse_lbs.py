"""Drop diagnostic fields (resid_max, resid_rms, resid_p95, method) from
reverse_lbs_results/*.npz to shrink storage.  Keeps only LBS-essential
fields: stream, level, fiber, origin_body, mid_body, insertion_body,
local_o, local_m, local_i, w_o, w_m, w_i.
"""
import glob
import os

import numpy as np

KEEP = {'stream', 'level', 'fiber',
        'origin_body', 'mid_body', 'insertion_body',
        'local_o', 'local_m', 'local_i',
        'w_o', 'w_m', 'w_i'}


def trim_one(path):
    d = np.load(path, allow_pickle=True)
    if 'records' not in d.files:
        return 0, 0
    recs_in = list(d['records'])
    recs_out = []
    for r in recs_in:
        recs_out.append({k: v for k, v in r.items() if k in KEEP})
    np.savez(path, records=np.array(recs_out, dtype=object))
    return len(recs_in), len(recs_out)


def main():
    files = sorted(glob.glob(
        '/home/jek/muscle_imitation_learning_study/reverse_lbs_results/*.npz'))
    before_total = 0
    after_total = 0
    for f in files:
        before_size = os.path.getsize(f)
        n_in, n_out = trim_one(f)
        after_size = os.path.getsize(f)
        before_total += before_size
        after_total += after_size
        print(f'  {os.path.basename(f)}: {n_in} recs, '
              f'{before_size/1024:.1f} KB -> {after_size/1024:.1f} KB')
    print(f'\nTotal: {before_total/1024/1024:.2f} MB -> {after_total/1024/1024:.2f} MB '
          f'({100*(1-after_total/before_total):.1f}% smaller)')


if __name__ == '__main__':
    main()
