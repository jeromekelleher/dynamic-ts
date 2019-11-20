"""
A prototype demonstrating how we'd build a forward simulator that
outputs a tskit tree sequence but that doesn't use tskit's data structures
internally.
"""
import random
import collections
import heapq
import sys

import tskit
import numpy as np


def overlapping_segments(segments):
    """
    Returns an iterator over the (left, right, X) tuples describing the
    distinct overlapping segments in the specified set.
    """
    S = sorted(segments, key=lambda x: x.left)
    n = len(S)
    # Insert a sentinel at the end for convenience.
    S.append(Segment(sys.float_info.max, 0, None))
    right = S[0].left
    X = []
    j = 0
    while j < n:
        # Remove any elements of X with right <= left
        left = right
        X = [x for x in X if x.right > left]
        if len(X) == 0:
            left = S[j].left
        while j < n and S[j].left == left:
            X.append(S[j])
            j += 1
        j -= 1
        right = min(x.right for x in X)
        right = min(right, S[j + 1].left)
        yield left, right, X
        j += 1

    while len(X) > 0:
        left = right
        X = [x for x in X if x.right > left]
        if len(X) > 0:
            right = min(x.right for x in X)
            yield left, right, X


class Segment(object):
    """
    An ancestral segment mapping a given individual to a half-open genomic
    interval [left, right).
    """
    def __init__(self, left, right, child):
        self.left = left
        self.right = right
        self.child = child

    def __repr__(self):
        return repr((self.left, self.right, self.child))


class OldIndividual(object):
    """
    Class representing a single individual that was alive at some time.
    """
    # This is just for debugging
    _next_index = 0
    def __init__(self, time, is_alive=True):
        self.time = time
        self.segments = []
        self.index = Individual._next_index
        Individual._next_index += 1

    def add_segment(self, left, right, child):
        u = Segment(left, right, child)
        # Clearly we'd want to do this more efficiently.
        squashed = False
        for v in self.segments:
            if v.right == left and v.child == child:
                v.right = right
                squashed = True
        if not squashed:
            self.segments.append(u)

    def __lt__(self, other):
        # We implement the < operator so that we can use the heapq directly.
        return self.time > other.time

    def __repr__(self):
        return f"Individual(index={self.index}, time={self.time})"


class OldSimulator(object):
    """
    Simple Wright-Fisher simulator using standard periodic simplify.
    """
    def __init__(self, population_size, sequence_length, death_proba=1.0, seed=None):
        self.population_size = population_size
        self.sequence_length = sequence_length
        self.death_proba = death_proba
        self.rng = random.Random(seed)
        self.time = 1
        self.alive = [OldIndividual(self.time) for _ in range(population_size)]
        self.dead = []

    def kill_individual(self, individual):
        """
        Append the specified individual to the dead list, keeping the
        list sorted by birth time.
        """
        # Using a heapq here for now, but there must be a better way to
        # do this. Surely most of the time this list will be almost sorted
        # and we can take advantage of that.
        heapq.heappush(self.dead, individual)

    def run_generation(self):
        """
        Implements a single generation.
        """
        self.time += 1
        replacements = []
        for j in range(self.population_size):
            if self.rng.random() < self.death_proba:
                left_parent = self.rng.choice(self.alive)
                right_parent = self.rng.choice(self.alive)
                # x = self.rng.uniform(0, self.sequence_length)
                # Using integers here just to make it easier to see what's going on.
                x = self.rng.randint(1, self.sequence_length - 1)
                assert 0 < x < self.sequence_length
                child = OldIndividual(self.time)
                replacements.append((j, child))
                left_parent.add_segment(0, x, child)
                right_parent.add_segment(x, self.sequence_length, child)
        for j, ind in replacements:
            self.kill_individual(self.alive[j])
            self.alive[j] = ind

    def run(self, num_generations, simplify_interval=1):
        for _ in range(num_generations):
            # self.simplify()
            # ts1 = self.export()

            self.run_generation()

#             ts2 = self.export()
#             ts3 = ts2.simplify()
#             print("GENERATION")
#             # labels = {u.id: f"({u.id})" if u.is_sample() else str(u.id) for u in ts.nodes()}
#             # print(ts.draw_text(node_labels=labels))
#             print(ts1.draw_text())
#             print(ts2.draw_text())
#             print(ts3.draw_text())

            # OK, here's the thinking. Let's assume that we keep the tree sequence fully
            # simplified at each generation. So, what we have at the end of a generation
            # is a fully simplified TS with some extra stuff grafted on near the leaves
            # (entirely at the leaves for non-overlapping gens). So, there *must* be a
            # more efficient algorithm for solving this problem than there is for solving
            # the general simplify problem.

            if self.time % simplify_interval == 0:
                # Debugging; check that we get the same answer as tskit.
                t1 = self.export().dump_tables()
                t1.simplify()
                self.simplify()
                t2 = self.export().dump_tables()
                print(len(t1.nodes), len(t2.nodes), ":", len(t1.edges), len(t2.edges))
                # print(t1.nodes)
                # print(t2.nodes)
                assert len(t1.nodes) == len(t2.nodes)
                # print(t1.edges)
                # print(t2.edges)
                # print("tskit")
                # print(t1.tree_sequence().draw_text())
                # print("local")
                # print(t2.tree_sequence().draw_text())
                assert len(t1.edges) == len(t2.edges)

    def simplify(self):
        """
        Garbage collects individuals and segments that can't be reached.
        """

        new_alive = []
        new_dead = []
        A = collections.defaultdict(list)
        ind_map = {}
        for sample in self.alive:
            new_ind = OldIndividual(sample.time)
            A[sample].append(Segment(0, self.sequence_length, new_ind))
            new_alive.append(new_ind)
            ind_map[sample] = new_ind

        # We have a really nice property here, in that we don't need to
        # consider the complications of overlapping having samples in
        # the internal loop at all. This doesn't seem to be the case in
        # general for simplify, so it would be good to understand why this
        # is the case.

        individuals = [heapq.heappop(self.dead) for i in range(len(self.dead))]

        # Visit the dead individuals in backwards in time.
        for input_ind in individuals:
            output_ind = None
            print("Consider", input_ind)

            S = []
            for e in input_ind.segments:
                # print("\t", len(A[e.child]))
                for x in A[e.child]:
                    if x.right > e.left and e.right > x.left:
                        y = Segment(
                            max(x.left, e.left), min(x.right, e.right), x.child)
                        S.append(y)

            for left, right, X in overlapping_segments(S):
                if len(X) == 1:
                    ancestry_node = X[0].child
                else:
                    if output_ind is None:
                        output_ind = OldIndividual(input_ind.time, False)
                        # A heapq is *definitely* wasteful here, because we're
                        # guaranteed to output these in sorted order. See the notes
                        # above in kill_individual for how we should be doing this
                        # better.
                        heapq.heappush(new_dead, output_ind)
                    ancestry_node = output_ind
                    for x in X:
                        output_ind.add_segment(left, right, x.child)
                alpha = Segment(left, right, ancestry_node)
                A[input_ind].append(alpha)
            print("A = ", A[input_ind])

        self.alive = new_alive
        self.dead = new_dead

    def export(self):
        """
        Exports the edges to a tskit tree sequence.
        """
        tables = tskit.TableCollection(self.sequence_length)
        node_map = {}
        for ind in reversed(self.alive):
            node_map[ind] = tables.nodes.add_row(
                flags=tskit.NODE_IS_SAMPLE,
                time=self.time - ind.time)
        for ind in sorted(self.dead, key=lambda x: -x.time):
            node_map[ind] = tables.nodes.add_row(
                flags=0, time=self.time - ind.time)
            mapped_parent = node_map[ind]
            for segment in sorted(
                    ind.segments, key=lambda x: (node_map[x.child], x.left)):
                mapped_child = node_map[segment.child]
                tables.edges.add_row(
                    left=segment.left, right=segment.right,
                    parent=mapped_parent, child=mapped_child)
        # print(tables)
        return tables.tree_sequence()


############################
# "New" implementation
############################

class Individual(object):
    """
    Class representing a single individual that was alive at some time.
    """
    # This is just for debugging
    _next_index = 0
    def __init__(self, time, is_alive=True):
        self.time = time
        self.is_alive = is_alive
        self.index = Individual._next_index
        Individual._next_index += 1

    def __repr__(self):
        return f"Individual(index={self.index}, time={self.time})"


class Edge(object):

    def __init__(self, left, right, parent, child):
        self.left = left
        self.right = right
        self.parent = parent
        self.child = child


class Simulator(object):
    """
    Simple Wright-Fisher simulator.
    """
    def __init__(self, population_size, sequence_length, death_proba=1.0, seed=None):
        self.population_size = population_size
        self.sequence_length = sequence_length
        self.death_proba = death_proba
        self.rng = random.Random(seed)
        self.time = 1
        self.alive = [Individual(self.time) for _ in range(population_size)]
        self.edges = []

    def add_ancestry(self, left, right, parent, child):
        """
        Adds an edge denoting the tranfer of ancestral material from the
        specified parent to the specified child.
        """
        self.edges.append(Edge(left, right, parent, child))

    def trees(self):
        """
        Returns an iterator over the trees. Each tree is a dictionary mapping
        an individual to its parent.
        """
        # Build the left and right indexes. Obvs keep these updated
        # dynamically for real thing.
        in_edges = iter(sorted(
            self.edges, key=lambda e: (e.left, -e.parent.time)))
        out_edges = iter(sorted(
            self.edges, key=lambda e: (e.right, e.parent.time)))
        sample_count = collections.Counter({ind: 1 for ind in self.alive})
        parent = {}

        in_edge = next(in_edges, None)
        out_edge = next(out_edges, None)
        while in_edge is not None:
            left = in_edge.left
            while out_edge is not None and out_edge.right == left:
                del parent[out_edge.child]
                u = out_edge.parent
                while u is not None:
                    sample_count[u] -= sample_count[out_edge.child]
                    u = parent.get(u, None)
                out_edge = next(out_edges, None)
            while in_edge is not None and in_edge.left == left:
                parent[in_edge.child] = in_edge.parent
                u = in_edge.parent
                while u is not None:
                    sample_count[u] += sample_count[in_edge.child]
                    u = parent.get(u, None)
                print("Done inserting edge", (in_edge.left, in_edge.right),
                        in_edge.child.index, "->", in_edge.parent.index,
                        "count = ", sample_count[in_edge.child])
                in_edge = next(in_edges, None)
            # Not quite right, will miss some cases where we have gaps.
            right = self.sequence_length if in_edge is None else in_edge.left
            yield (left, right), parent, sample_count



    def run_generation(self):
        """
        Implements a single generation.
        """
        self.time += 1
        for interval, tree, sample_count in self.trees():
            print(interval, tree, sample_count)
        replacements = []
        for j in range(self.population_size):
            if self.rng.random() < self.death_proba:
                left_parent = self.rng.choice(self.alive)
                right_parent = self.rng.choice(self.alive)
                # x = self.rng.uniform(0, self.sequence_length)
                # Using integers here just to make it easier to see what's going on.
                x = self.rng.randint(1, self.sequence_length - 1)
                assert 0 < x < self.sequence_length
                child = Individual(self.time)
                replacements.append((j, child))
                self.add_ancestry(0, x, left_parent, child)
                self.add_ancestry(x, self.sequence_length, right_parent, child)
        for j, ind in replacements:
            self.alive[j].is_alive = False
            self.alive[j] = ind

    def run(self, num_generations):
        for _ in range(num_generations):
            self.run_generation()
            print("Generation", self.time)
            ts = self.export()
            labels = {u.id: f"({u.id})" if u.is_sample() else str(u.id) for u in ts.nodes()}
            print(ts.draw_text(node_labels=labels))


    def export(self):
        """
        Exports the edges to a tskit tree sequence.
        """
        tables = tskit.TableCollection(self.sequence_length)
        # Map the individuals to their indexes to make it debug easier.
        individuals = {}
        for edge in self.edges:
            individuals[edge.child.index] = edge.child
            individuals[edge.parent.index] = edge.parent
        print(individuals)
        for j in range(max(individuals.keys()) + 1):
            if j in individuals:
                ind = individuals[j]
                # print("adding", ind)
                ret = tables.nodes.add_row(
                    flags=tskit.NODE_IS_SAMPLE if ind.is_alive else 0,
                    time=self.time - ind.time)
                assert ret == j
            else:
                tables.nodes.add_row(0, 0)
            # assert node_map[ind] == ind.index

        for edge in self.edges:
            # for ind in edge.parent, edge.child:
            #     if ind not in node_map:
            #         node_map[ind] = tables.nodes.add_row(
            #             flags=tskit.NODE_IS_SAMPLE if ind.is_alive else 0,
            #             time=self.time - ind.time)
            tables.edges.add_row(
                left=edge.left, right=edge.right,
                parent=edge.parent.index, child=edge.child.index)
        tables.sort()
        return tables.tree_sequence()




def main():
    # for seed in range(1, 10000):
    # for seed in [68]:
        # print(seed)

    seed = 1
    sim = OldSimulator(15, 20, death_proba=0.1, seed=seed)
    sim.run(20, 10)
    # ts = sim.export()
    # print(ts.draw_text())
    # ts_simplify = ts.simplify()
    # print(ts_simplify.draw_text())


if __name__ == "__main__":
    main()
