"""
Prototype forward simulator where we simplify continuously throughout
the simulation.
"""
import random
import collections
import heapq
import sys

import tskit
import sortedcollections


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


class Individual(object):
    """
    Class representing a single individual that was alive at some time.
    """
    # This is just for debugging
    _next_index = 0
    def __init__(self, time, is_alive=True):
        self.time = time
        self.segments = []
        self.index = Individual._next_index
        self.is_alive = is_alive
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

    def __repr__(self):
        return f"Individual(index={self.index}, time={self.time}, is_alive={self.is_alive})"

class Interval(object):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Interval(left={self.left}, right={self.right})"


class Simulator(object):
    """
    Simple Wright-Fisher simulator using standard periodic simplify.
    """
    def __init__(self, population_size, sequence_length, death_proba=1.0, seed=None):
        self.population_size = population_size
        self.sequence_length = sequence_length
        self.death_proba = death_proba
        self.rng = random.Random(seed)
        self.time = 1
        self.individuals = [Individual(self.time) for _ in range(population_size)]

    def kill_individual(self, individual):
        """
        Kill the specified individual.
        """
        individual.is_alive = False

    def run_generation(self):
        """
        Implements a single generation.
        """
        self.time += 1
        replacements = []
        alive_indexes = [i for i, j in enumerate(self.individuals) if j.is_alive]
        for j in alive_indexes:
            if self.rng.random() < self.death_proba:
                left_parent = self.individuals[self.rng.choice(alive_indexes)]
                right_parent = self.individuals[self.rng.choice(alive_indexes)]
                # x = self.rng.uniform(0, self.sequence_length)
                # Using integers here just to make it easier to see what's going on.
                x = self.rng.randint(1, self.sequence_length - 1)
                assert 0 < x < self.sequence_length
                child = Individual(self.time)
                replacements.append((j, child))
                left_parent.add_segment(0, x, child)
                right_parent.add_segment(x, self.sequence_length, child)
        for j, ind in replacements:
            self.kill_individual(self.individuals[j])
            self.individuals.append(ind)
        # Ultimately we should be passing the fact that only these
        # individuals are new, and should be able to update the simplify
        # data structures in place. For now, let's get it working where
        # we update the structures in-place rather than replacing everything.
        self.check_state()
        self.simplify()
        self.check_state()
        # NOTE: expensive check
        try:
            self.export()
        except:
            print(self.time)
            raise


    def run(self, num_generations, simplify_interval=1):
        for _ in range(num_generations):
            self.run_generation()
            # print(len(self.dead))


    def simplify(self):
        """
        Garbage collects individuals and segments that can't be reached.
        """
        # print("Simplify")
        # ts = self.export()
        # print(ts.draw_text())

        A = collections.defaultdict(list)
        for ind in self.individuals:
            # All alive individuals map to themselves.
            if ind.is_alive:
                A[ind].append(Segment(0, self.sequence_length, ind))

        # Visit the individuals in reverse order of birth time (i.e.,
        # backwards in time from the present)
        for ind in reversed(self.individuals):
            S = []
            for e in ind.segments:
                # print("\t", e)
                for x in A[e.child]:
                    if x.right > e.left and e.right > x.left:
                        y = Segment(
                            max(x.left, e.left), min(x.right, e.right), x.child)
                        S.append(y)

            ind.segments.clear()
            for left, right, X in overlapping_segments(S):
                if len(X) == 1:
                    # If we have no coalescence on this interval, map the
                    # ancestry to the child, getting rid of this unary node.
                    mapped_ind = X[0].child
                else:
                    # If we have coalescences, then we keep this individual
                    # and add back in the appropriate edges.
                    mapped_ind = ind
                    for x in X:
                        ind.add_segment(left, right, x.child)
                # If an individual is alive, we have already mapped it
                # to itself in the setup of A, above, and so we
                # prevent an additional mapping here.
                if not ind.is_alive:
                    A[ind].append(Segment(left, right, mapped_ind))
            # If this individual has no children, and
            # is not an alive individual, then we don't need
            # it anymore and it can be garbage collected.
            # NOTE: us a NULL value to trigger GC
            if len(ind.segments) == 0 and not ind.is_alive:
                ind.is_alive = None
        before = len(self.individuals)
        self.individuals = [i for i in self.individuals if i.is_alive is not None]

    def check_state(self):
        num_alive = 0
        for i in self.individuals:
            if i.is_alive is True:
                num_alive += 1
        assert num_alive == self.population_size
        # Every segment that we refer to should be in dead or alive.
        for ind in self.individuals:
            segs = sorted(ind.segments, key=lambda x: x.left)
            # print("checking", ind)
            # for seg in segs:
            #     print("\t", seg)
            for seg in segs:
                assert seg.child in self.individuals, f"{self.time} {seg.child}"


    # def export(self):
    #     """
    #     Exports the edges to a tskit tree sequence.
    #     """
    #     tables = tskit.TableCollection(self.sequence_length)
    #     node_map = {}
    #     for ind in reversed(self.alive):
    #         node_map[ind] = tables.nodes.add_row(
    #             flags=tskit.NODE_IS_SAMPLE,
    #             time=self.time - ind.time)
    #     for ind in reversed(self.dead):
    #         node_map[ind] = tables.nodes.add_row(
    #             flags=0, time=self.time - ind.time)
    #         mapped_parent = node_map[ind]
    #         for segment in sorted(
    #                 ind.segments, key=lambda x: (node_map[x.child], x.left)):
    #             mapped_child = node_map[segment.child]
    #             tables.edges.add_row(
    #                 left=segment.left, right=segment.right,
    #                 parent=mapped_parent, child=mapped_child)
    #     # print(tables)
    #     return tables.tree_sequence()


    def export(self):
        """
        Exports the edges to a tskit tree sequence.

        NOTE: the individuals themselves are sorted by birth order.
        The segments w/in an individual are/should be/maybe
        quite close to sorted.  Thus, a full table sort
        is probably wasteful and we sort segments w/in
        individuals instead, which can be trivially
        parallelized across individuals.
        """
        tables = tskit.TableCollection(self.sequence_length)
        # Map the individuals to their indexes to make debug easier.
        individuals = {ind.index: j for j, ind in enumerate(reversed(self.individuals))}
        for ind in reversed(self.individuals):
            # print("adding", ind)
            ret = tables.nodes.add_row(
                flags=tskit.NODE_IS_SAMPLE if ind.is_alive is True else 0,
                time=self.time - ind.time)

        for ind in reversed(self.individuals):
            segments = sorted(
                ind.segments,
                key = lambda x: (-x.child.time, individuals[x.child.index], x.left))
            for seg in segments:
                tables.edges.add_row(
                    left=seg.left, right=seg.right,
                    parent=individuals[ind.index], child=individuals[seg.child.index])
        # print(tables)
        return tables.tree_sequence()


def main():
    seed = 1
    # sim = Simulator(4, 5, death_proba=1.0, seed=seed)
    sim = Simulator(400, 5, death_proba=0.5, seed=seed)
    sim.run(100)
    # ts = sim.export()
    # print(ts.draw_text())
    # ts_simplify = ts.simplify()
    # print(ts_simplify.draw_text())


if __name__ == "__main__":
    main()
