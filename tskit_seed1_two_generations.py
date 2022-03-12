import tskit

N = 4
L = 5
tables = tskit.TableCollection(L)


parents = []

for t in range(3):
    for _ in range(N):
        p = tables.nodes.add_row(0, 3 - t - 1)
        parents.append(p)

x = [1, 4, 1, 1]  # xover positions
parents = [(0, 2), (3, 3), (0, 3), (3, 3)]
base = N

for i in range(N):
    tables.edges.add_row(0, x[i], parents[i][0], base + i)
    tables.edges.add_row(x[i], L, parents[i][1], base + i)

x = [1, 1, 2, 4]  # xover positions
parents = [
    (base + 2, base + 1),
    (base + 0, base + 0),
    (base + 0, base + 3),
    (base + 0, base + 1),
]
base = 2 * N

print(parents)

for i in range(N):
    tables.edges.add_row(0, x[i], parents[i][0], base + i)
    tables.edges.add_row(x[i], L, parents[i][1], base + i)

tables.sort()

samples = [base + i for i in range(N)]
idmap = tables.simplify(samples)

print(samples)
print(idmap)

ts = tables.tree_sequence()

print(ts.draw_text())
