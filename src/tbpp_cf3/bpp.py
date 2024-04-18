from dataclasses import dataclass
from collections import Counter
import gurobipy as gp

__all__ = ['Instance', 'build']


@dataclass
class Instance:
    c: list[int]
    cap: int

    @property
    def n(self):
        return len(self.c)

    def sorted(self):
        return Instance(sorted(self.c, reverse=True), self.cap)


def build(inst: Instance):
    cnt = Counter(inst.c)
    its = sorted(cnt.items(), reverse=True)

    verts = {0}
    item_arcs = []
    arcs = set()

    for ci, bi in its:
        arcs_i = set()
        for u in sorted(verts, reverse=True):
            for _ in range(bi):
                v = u + ci
                if v > inst.cap:
                    break
                verts.add(v)
                arcs_i.add((u, v))
                u = v
        item_arcs.append(arcs_i)
        arcs |= arcs_i

    m = gp.Model()
    x = m.addVars(arcs, name='x', vtype=gp.GRB.INTEGER)

    m.addConstrs(
        (
            gp.quicksum(x[arc] for arc in item_arcs[i]) >= bi
            for i, (ci, bi) in enumerate(its)
        ),
        name='sat',
    )
    m.addConstrs(
        (x.sum('*', u) >= x.sum(u, '*') for u in verts if u > 0),
        name='conserve',
    )

    m.setObjective(x.sum(0, '*'))
    return m
