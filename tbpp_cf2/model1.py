from itertools import chain
import gurobipy as gp
from . import InstanceTBPPFU
from .util import pairwise, compute_conflict_cliques

__all__ = ['build']


def build(
    inst: InstanceTBPPFU,
    lb_servers: int = 0, ub_servers: int = 0,
    mods: set[str] = {'conflicts'},
):

    assert mods <= {'conflicts', 'wy', 'continuous_w'}

    # idx set of items (jobs)
    idx_i = range(inst.n)
    # idx set of bins (servers)
    n_servers = inst.n if ub_servers == 0 else ub_servers
    idx_k = range(n_servers)

    ts_k = {k: set(inst.s[k:]) for k in idx_k}
    te_k = {k: set(inst.e[k:]) for k in idx_k}
    t_k = {k: sorted(ts_k[k] | te_k[k]) for k in idx_k}

    tsnd_k = {
        k: {
            t0 for t0, t1 in pairwise(t_k[k])
            if t0 in ts_k[k] and t1 in te_k[k]
        }
        for k in idx_k
    }

    last_ts_k = {k: max(ts_k[k]) for k in idx_k}
    te_k = {k: {
        inst.e[i]
        for i in idx_i
        if i >= k and inst.e[i] < last_ts_k[k]
    } for k in idx_k}
    t_k = {k: sorted(ts_k[k] | te_k[k]) for k in idx_k}

    m = gp.Model()

    # add variables
    x = m.addVars(
        [(i, k) for i in idx_i for k in idx_k if i >= k],
        vtype=gp.GRB.BINARY, name='x'
    )
    z = m.addVars(
        idx_k,
        obj=1, vtype=gp.GRB.BINARY, name='z'
    )
    y = m.addVars(
        ((t, k) for k in idx_k for t in t_k[k]),
        lb=0.0, ub=1.0, vtype=gp.GRB.BINARY, name='y'
    )
    w_type = gp.GRB.CONTINUOUS if 'continuous_w' in mods else gp.GRB.BINARY
    w = m.addVars(
        ((t, k) for k in idx_k for t in ts_k[k]),
        obj=inst.gamma,
        lb=0.0, ub=1.0, vtype=w_type, name='w'
    )

    m._servers = z.sum()
    m._fireups = w.sum()

    # capacity constraint and activity of server
    m.addConstrs((
        gp.quicksum(
            inst.c[i] * x[i, k]
            for i in idx_i if i >= k and inst.s[i] <= t and t < inst.e[i]
        ) <= inst.cap * y[t, k]
        for k in idx_k
        for t in tsnd_k[k]
    ), name='on')

    m.addConstrs((
        gp.quicksum(
            x[i, k]
            for i in idx_i if i >= k and inst.s[i] <= t and t < inst.e[i]
        ) >= y[t, k]
        for k in idx_k
        for t in te_k[k]
    ), name='off')

    # exactly one server per job
    m.addConstrs((
        x.sum(i, '*') == 1
        for i in idx_i
    ), name='assign')

    # coupling of x and y
    m.addConstrs((
        x[i, k] <= y[inst.s[i], k]
        for k in idx_k
        for i in idx_i if k <= i
    ), name='act')

    # coupling of y and z
    m.addConstrs((
        y[t, k] <= z[k]
        for k in idx_k
        for t in tsnd_k[0] & ts_k[k]
    ), name='use_y')

    # coupling of y and w
    m.addConstrs((
        y[t, k] <= w[t, k] + (0.0 if tp == 's' else y[tp, k])
        for k in idx_k
        for tp, t in pairwise(chain(['s'], t_k[k])) if t in ts_k[k]
    ), name='fireup')

    if 'wy' in mods:
        m.addConstrs((
            w[t, k] <= y[t, k]
            for k in idx_k
            for tp, t in pairwise(t_k[k]) if t in ts_k[k]
        ), name='wy_on')
        m.addConstrs((
            w[t, k] <= 1 - y[tp, k]
            for k in idx_k
            for tp, t in pairwise(t_k[k]) if t in ts_k[k]
        ), name='wy_off')

    # use bound on server count
    if lb_servers > 0:
        m.addConstrs(
            (z[k] == 1 for k in range(lb_servers)),
            name='lb_server'
        )

    # break symmetry
    m.addConstrs((
        z[k] >= z[k + 1]
        for k in idx_k[lb_servers:-1]
    ), name='break_symmetry')

    # add VIs
    m.addConstrs((
        z[k] <= w.sum('*', k)
        for k in idx_k
    ), name='server_fireup')

    if 'conflicts' in mods:
        ccs = compute_conflict_cliques(inst, n_servers)
        for k, cs_k in enumerate(ccs):
            for i, cs in cs_k.items():
                for l, c in enumerate(cs):
                    sc = max(inst.s[j] for j in c)
                    m.addConstr(
                        gp.quicksum(x[j, k] for j in c) <= y[sc, k],
                        name=f'conflict[{k},{i},{l}]'
                    )

    return m
