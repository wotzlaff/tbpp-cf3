import time
import os
from functools import partial
from gurobipy import GRB
import tbpp_cf3


def read_bounds():
    res = {}
    with open('./data/lb_servers_1.csv') as f:
        for line in f.readlines()[1:]:
            inst_name, value = line.split(',')
            res[inst_name] = int(value)
    return res


def main():
    models = [
        (
            'model1n_bpp_wy_heu_s',
            partial(tbpp_cf3.model1.build, mods={'conflicts', 'bpp', 'wy'}),
        ),
        (
            'model1ts_bpp_wy_heu_s',
            partial(
                tbpp_cf3.model1.build,
                mods={'all_ts', 'conflicts', 'bpp', 'wy'},
            ),
        ),
    ]
    os.makedirs('./logs/mip', exist_ok=True)

    root = './data/tbpp-instances/data/a1'
    groups = tbpp_cf3.data.format1.get_groups(root)

    lb_servers_dict = read_bounds()
    gamma = 1.0

    for group in groups:
        n, t, kind = group
        group_name = f'n{n} t{t} {kind}'
        print(group_name)
        for model_name, build in models:
            print(model_name)
            log_path = f'./logs/mip/{group_name}_{model_name}.log'
            # if os.path.exists(log_path):
            #     continue
            with open(log_path, 'w') as fout:
                fout.write(
                    'name\tNumVars\tNumConstrs\tNumNZs\tdt_model\tdt_solve\tdt_relax\tm.ObjVal\tmrel.ObjVal\toptimal\tservers\tstartups\n'
                )
                fout.flush()

                for inst_name, inst in tbpp_cf3.data.format1.read_instances(
                    root, group_name
                ):
                    print(inst_name)

                    # lift instance
                    inst = tbpp_cf3.lift(inst).sorted()
                    inst = tbpp_cf3.InstanceTBPPFU.extend(inst, gamma=gamma)
                    lb_servers = lb_servers_dict[inst_name]
                    alloc = tbpp_cf3.heuristic.best_look_ahead(
                        inst, {1, 2, 3, 5, 10, 20, inst.n // 4, inst.n // 2, inst.n}
                    )
                    value = inst.compute_value(alloc)
                    ub_servers = int(value / (1.0 + inst.gamma) + 1e-5)
                    t0 = time.time()
                    m = build(
                        inst,
                        lb_servers=lb_servers,
                        ub_servers=ub_servers,
                    )

                    if model_name.endswith('_s'):
                        m._set_start(alloc)

                    m.setParam('OutputFlag', 0)
                    m.setParam('TimeLimit', 1800)
                    m.setParam('Method', 3)
                    dt_model = time.time() - t0
                    if m.Status == GRB.INTERRUPTED:
                        print('interrupted')
                        return

                    t0 = time.time()
                    m.update()
                    m.optimize()
                    dt_solve = time.time() - t0

                    optimal = m.Status == GRB.OPTIMAL
                    servers = 0
                    fireups = 0
                    if optimal:
                        servers = round(m._servers.getValue())
                        fireups = round(m._fireups.getValue())

                    t0 = time.time()
                    mrel = m.relax()
                    mrel.optimize()
                    dt_relax = time.time() - t0

                    fout.write(
                        f'{inst_name}\t{m.NumVars}\t{m.NumConstrs}\t{m.NumNZs}\t{dt_model}\t{dt_solve}\t{dt_relax}\t{m.ObjVal}\t{mrel.ObjVal}\t{optimal}\t{servers}\t{fireups}\n'
                    )
                    fout.flush()


if __name__ == '__main__':
    main()
