import tskit

tc = tskit.TableCollection(5)

p = []
c = []

for i in range(4):
    p.append(tc.nodes.add_row(time=1))
for i in range(4):
    c.append(tc.nodes.add_row(time=0))

parents = [(0, 0), (1, 2), (0, 1), (2, 3)]
x = [1, 3, 4, 3]  # xover positions

for i in range(4):
    tc.edges.add_row(0, x[i], parents[i][0], c[i])
    tc.edges.add_row(x[i], 5, parents[i][1], c[i])

tc.sort()

idmap = tc.simplify(c)

print(tc)

for i in c:
    print(i, idmap[i])
