"""
Prototype forward simulator where we simplify inline, using the Individual
structures directly.
"""
from dataclasses import dataclass
import random
import collections
import heapq
import sys
from typing import List, Tuple

import numpy as np
import tskit

# Notes: this is not working yet. The idea is that we should be able to
# maintain the data structure needed for simplify dynamically as (a)
# ancestry segments are added when samples are added as children of
# the individuals and (b) as individuals die. In the case of (a), we
# can have coalescences occur or not, as new children are added to a
# parent. That's easy to track, by looking at the overlapping segments
# within the children. In (b) when an individual dies, this means a
# loss of ancestral material up the trees. Because we're keeping the
# list of parents for each individual, we should be able to propagate
# this loss up the trees in a simplify-esque way. If we do it one-by-one,
# then the changes to the data structures should be easy enough to reason
# about.

# The goal of all this is to avoid the O(N) cost of visiting all N
# reachable individuals each generation. By keeping changes local to
# the actual losses and gains of ancestral material, we should be able
# to keep this to something more like log(N). This seems pointless if
# we're going to be iterating over all the trees each generation though.
# Perhaps we can work around this by keeping a tree for each selected
# site 'live' in memory, and update it as well as edges are added and
# lost. If we use hash tables for the trees, then presumably this isn't
# too much memory.

# Not sure if any of this will work, but that's the thinking anyway.


def assert_non_overlapping(segments):
    for j in range(1, len(segments)):
        x = segments[j - 1]
        y = segments[j]
        assert x.right <= y.left, f"{segments[j-1]} {segments[j]}"


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


def propagate_upwards(ind):
    pass
    # This isn't working.
    # print("PROPAGATE", ind)
    stack = [ind]
    while len(stack) > 0:
        ind = stack.pop()
        # print("\t", ind)
        # ind.print_state()
        # We're visting everything here at the moment, but we don't need to.
        # We should only have to visit the parents for which we have ancestral
        # segents, and so the areas of the graph we traverse should be
        # quickly localised.
        # print("before")
        # ind.print_state()
        # print(f"updating {ind}")
        ind.update_ancestry()
        # print("after")
        # ind.print_state()
        for parent in ind.parents:
            assert parent.time < ind.time
            stack.append(parent)


def record_inheritance(left, right, parent, child):
    """
    Record the inheritances of genetic material from parent to child over the
    from coordinate left to right.
    """
    child.parents.add(parent)
    parent.add_child_segment(child, left, right)
    # print("record", child, parent, child.parents)


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

    def __eq__(self, o):
        return self.left == o.left and self.right == o.right and self.child == o.child


class Individual(object):
    """
    Class representing a single individual that was alive at some time.
    """

    def __init__(self, time, index, is_alive=True):
        self.time = time
        # The direct children segments of this individual. For each unique
        # child, we keep a list of the ancestral segments that it has inherited.
        self.children = collections.defaultdict(list)
        # The ancestry mapping for this individual, as derived from the children
        # segments. If this individual is currently alive, this is a single
        # segment mapping to itself. If not, each segment records the closest
        # tree node (i.e., where there is a sample or a coalescence). Coalescences
        # occur when overlaps happen in the children.
        self.ancestry = []
        # The set of parent Individuals from which this individual has inherited
        # genetic material.
        # FIXME: this comment is not quite right.  For new births, it is the
        # parents from which segments are inherited.  For older nodes, it
        # may or may not refer to parents transmitting unary edges to self.
        self.parents = set()
        self.is_alive = is_alive
        self.index = index

    def __repr__(self):
        return f"Individual(index={self.index}, time={self.time}, is_alive={self.is_alive})"

    def print_state(self):
        print(repr(self))
        print("children = ")
        for child, intervals in self.children.items():
            print("\t", child, "->", [(x.left, x.right) for x in intervals])
        print("ancestry = ")
        for i in self.ancestry:
            print(i)
        print("parents = ", self.parents)

    def add_child_segment(self, child, left, right):
        """
        Adds the ancestry for the specified child over the specified interval.
        """
        self.children[child].append(Segment(left, right, None))

    def remove_sample_mapping(self, sequence_length):
        """
        Alive ("sample") individuals (nodes) contain a segment [0, sequence_length), self.
        When such individuals die, this segment needs to be removed.
        """

        def not_a_sample_map(seg, sequence_length) -> bool:
            assert seg.child is not None
            if seg.child.index != self.index:
                return True
            if not (seg.left == 0 and seg.right == sequence_length):
                return True
            return False

        self.ancestry = [
            i
            for i in filter(
                lambda seg: not_a_sample_map(seg, sequence_length),
                self.ancestry,
            )
        ]

    def intersecting_ancestry(self):
        """
        Returns the list of intervals over which this interval's children intersect
        with those children's ancestry segments.
        """
        S = []
        for child, intervals in self.children.items():
            for e in intervals:
                for x in child.ancestry:
                    if x.right > e.left and e.right > x.left:
                        y = Segment(max(x.left, e.left), min(x.right, e.right), child)
                        S.append(y)
        return S

    def update_ancestry(self):
        # FIXME: for overlapping generations to work, we need
        # to take some hints from standard simplification:
        # We need to clear out the existing ancestry for alive nodes, and
        # then track the right-most coordinate added into
        # the new ancestry.  If that value is < L at the end,
        # we need to fill in the overhang from last_right -> L
        S = self.intersecting_ancestry()

        print(f"START {self.index}")
        print(f"S = {S}")
        for c in self.children.keys():
            print(f"child {c} has ancestry {c.ancestry}, children {c.children}")
        self.print_state()

        # Do not "bulk" clear: we need to know
        # if current children are no longer
        # children when done.
        # TODO: investigate possibility of
        # removing self from parents of all children
        # as an alternative strategy
        for c, s in self.children.items():
            s.clear()

        for left, right, X in overlapping_segments(S):
            # print(left, right, X)
            if len(X) == 1:
                mapped_ind = X[0].child
                print("unary", mapped_ind, X[0])
                # unary_mapped_ind.append(mapped_ind)
                if self in mapped_ind.parents:
                    # need to guard against unary
                    # edges to the right of coalescences
                    print("YES")
                    if mapped_ind not in self.children:
                        mapped_ind.parents.remove(self)
                    # if mapped_ind.is_alive and self.is_alive:
                    if self.is_alive:
                        if mapped_ind.is_alive:
                            print("SEG ADDING", mapped_ind)
                            self.add_child_segment(mapped_ind, left, right)
                        else:
                            print(
                                "NEED TO PROCESS UNARY THRU DEAD UNARY NODE", mapped_ind
                            )
                            for a in mapped_ind.ancestry:
                                if a.right > left and right > a.left:
                                    mapped_ind = a.child
                                    if self not in mapped_ind.parents:
                                        mapped_ind.parents.add(self)
                                    self.add_child_segment(
                                        mapped_ind,
                                        max(left, a.left),
                                        min(right, a.right),
                                    )
            else:
                output_mappings = set()
                for x in X:
                    if len(x.child.children) > 0 or x.child.is_alive:
                        for a in x.child.ancestry:
                            if a.right > left and right > a.left:
                                if a.child is not x.child:
                                    # NOTE: TODO: FIXME: AHA?!?!?!?
                                    print(
                                        "COAL TO A UNARY SEGMENT",
                                        self,
                                        "->",
                                        x.child,
                                        a.child,
                                        left,
                                        right,
                                        a.left,
                                        a.right,
                                    )
                                    x.left = max(left, a.left)
                                    x.right = min(right, a.right)
                                    x.child = a.child
                        output_mappings.add(x.child)
                        self.add_child_segment(x.child, left, right)
                        if self not in x.child.parents:
                            x.child.parents.add(self)
                    else:
                        print("TRAVERSING DOWN A UNARY", x.child)
                        for a in x.child.ancestry:
                            if a.right > left and right > a.left:
                                self.add_child_segment(
                                    a.child, max(left, a.left), min(right, a.right)
                                )
                                assert a.child in self.children
                                a.child.parents.add(self)
                                output_mappings.add(a.child)
                                break
                if len(output_mappings) > 1:
                    mapped_ind = self
                else:
                    assert len(output_mappings) == 1
                    mapped_ind = [output_mappings.pop()]
                if mapped_ind in self.children:
                    # FIXME: this is a really annoyting gotcha:
                    # the defaultdict(list) adds [] to self.children
                    # if it does not exist
                    assert_non_overlapping(self.children[mapped_ind])
            # If an individual is alive it always has ancestry over the
            # full segment, so we don't overwrite this.
            if not self.is_alive:
                seg = Segment(left, right, mapped_ind)
                new_segment = True
                for i in self.ancestry:
                    if i.right > seg.left and seg.right > i.left:
                        i.left = max(i.left, seg.left)
                        i.right = min(i.right, seg.right)
                        i.child = mapped_ind
                        new_segment = False
                if new_segment:
                    self.ancestry.append(seg)

        if not self.is_alive:
            self.ancestry = sorted(
                [i for i in self.ancestry if i.child is not None], key=lambda x: x.left
            )

        assert_non_overlapping(self.ancestry)

        if not self.is_alive:
            for c, segments in self.children.items():
                if c is not self:
                    if len(segments) > 0:
                        assert self in c.parents, f"{self} {c} {self.ancestry}"
        new_children = collections.defaultdict(list)
        for c, s in self.children.items():
            if len(s) == 0:
                # there are no coalescences from parent -> child
                if self in c.parents:  # and not self.is_alive:
                    c.parents.remove(self)
            else:
                new_children[c] = s
        self.children = new_children
        for a in self.ancestry:
            if a.child in self.children and a.child is not self:
                assert self in a.child.parents, f"{self} {a} {self.ancestry}"
        print("DONE")
        self.print_state()
        print("OUT")


@dataclass
class TransmissionInfo:
    parent: Individual
    child: Individual
    left: int
    right: int


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
        self.population = [Individual(self.time, i) for i in range(population_size)]
        self.next_individual_index = population_size
        self.transmissions: List[TransmissionInfo] = []

        # Everyone starts out alive, so has to map to self
        # (Overlapping generations fails fast if this isn't done)
        for i in self.population:
            assert len(i.ancestry) == 0
            assert i.index < population_size
            i.ancestry.append(Segment(0, sequence_length, i))

    def run_generation(self):
        """
        Implements a single generation.
        """
        self.time += 1
        replacements = []
        for j in range(self.population_size):
            if self.rng.random() < self.death_proba:
                left_parent = self.rng.choice(self.population)
                right_parent = self.rng.choice(self.population)
                # left_parent_index = -1
                # for k in range(self.population_size):
                #     if left_parent is self.population[k]:
                #         left_parent_index = k
                #         break
                # right_parent_index = -1
                # for k in range(self.population_size):
                #     if right_parent is self.population[k]:
                #         right_parent_index = k
                #         break
                # Using integers here just to make debugging easier
                x = self.rng.randint(1, self.sequence_length - 1)
                assert 0 < x < self.sequence_length
                # print(
                #     "BIRTH:",
                #     left_parent,
                #     left_parent_index,
                #     right_parent,
                #     right_parent_index,
                #     j,
                #     x,
                # )
                child = Individual(self.time, self.next_individual_index)
                self.next_individual_index += 1
                child.ancestry = [Segment(0, self.sequence_length, child)]
                replacements.append((j, child))
                record_inheritance(0, x, left_parent, child)
                record_inheritance(x, self.sequence_length, right_parent, child)
                self.transmissions.append(
                    TransmissionInfo(
                        left_parent,
                        child,
                        0,
                        x,
                    )
                )
                self.transmissions.append(
                    TransmissionInfo(
                        right_parent,
                        child,
                        x,
                        self.sequence_length,
                    )
                )

        # First propagate the loss of the ancestral material from the newly dead
        # print("pdead")
        for j, ind in replacements:
            dead = self.population[j]
            dead.is_alive = False
            dead.remove_sample_mapping(sequence_length=self.sequence_length)
            print(f"propagating death {dead}")
            propagate_upwards(dead)
            self.population[j] = ind
        for _, ind in replacements:
            print(f"propagating birth {ind}")
            propagate_upwards(ind)
        self.check_state()

    def check_state(self):
        reachable = self.all_reachable()
        for ind in reachable:
            if ind.is_alive:
                assert len(ind.ancestry) == 1
                x = ind.ancestry[0]
                assert x.left == 0
                assert x.right == self.sequence_length
                assert x.child == ind
            else:
                assert_non_overlapping(ind.ancestry)
            for child, segments in ind.children.items():
                if child is not ind:
                    assert child in reachable, f"{child} {ind} {ind.children}"
                assert_non_overlapping(segments)
                # NOTE: this happens b/c as a side-effect of defaultdict
                if child is not ind:
                    assert ind in child.parents, f"{ind} {child}"
        for parent in ind.parents:
            assert parent in reachable
            if ind not in parent.children:
                print("the failing parent is")
                parent.print_state()
                print("done w/failing parent")
            assert ind in parent.children, f"{ind} {parent}"

    def run(self, num_generations, simplify_interval=1):
        for _ in range(num_generations):
            self.run_generation()

    def make_samples_list_for_tskit(self, node_map) -> List[int]:
        rv = []
        for i in self.transmissions:
            for j in [i.parent, i.child]:
                if j.is_alive and node_map[j.index] not in rv:
                    rv.append(node_map[j.index])
        return sorted(rv)

    def get_alive_node_indexes_and_times(self) -> List[Tuple[int, int]]:
        indexes = set()
        rv = []
        for i in self.transmissions:
            for j in [i.parent, i.child]:
                if j.is_alive and j.index not in indexes:
                    rv.append((j.index, j.time))
                    indexes.add(j.index)

        return rv

    def convert_transmissions_to_tables(
        self,
    ) -> Tuple[tskit.TableCollection, List[int]]:
        """
        Take the raw transmission info and make it into
        an unsimplified TableCollection for an independent
        look at the simulated topologies.
        """
        tables = tskit.TableCollection(self.sequence_length)

        # There's probably a faster way to do all this...
        node_data = {}
        max_time = -1
        for i in self.transmissions:
            for n in [i.parent, i.child]:
                if n.index not in node_data:
                    node_data[n.index] = n.time
                    max_time = max(max_time, n.time)

        alive_nodes = self.get_alive_node_indexes_and_times()

        max_alive_node_time = max([i[1] for i in alive_nodes])

        assert max_time == max_alive_node_time

        node_map = {}
        for i, t in node_data.items():
            x = tables.nodes.add_row(0, -(t - max_time))
            node_map[i] = x

        samples = self.make_samples_list_for_tskit(node_map)

        double_check = sorted([node_map[i[0]] for i in alive_nodes])
        assert samples == double_check, f"{samples} {double_check}"

        for t in self.transmissions:
            tables.edges.add_row(
                t.left, t.right, node_map[t.parent.index], node_map[t.child.index]
            )

        tables.sort()

        return tables, samples, node_map

    def all_reachable(self):
        """
        Returns the set of all individuals reachable from the current populations.
        """
        individuals = set()
        for ind in self.population:
            stack = [ind]
            while len(stack) > 0:
                ind = stack.pop()
                individuals.add(ind)
                for parent in ind.parents:
                    if parent not in individuals:
                        stack.append(parent)
        return individuals

    def export(self):
        """
        Exports the edges to a tskit tree sequence.
        """
        tables = tskit.TableCollection(self.sequence_length)
        # Map the individuals to their indexes to make debug easier.
        # THIS IS A TERRIBLE IDEA!!!
        sorted_individuals = sorted(self.all_reachable(), key=lambda x: x.index)
        next_ind = 0
        for ind in sorted_individuals:
            while ind.index != next_ind:
                # Add in a padding node.
                tables.nodes.add_row(flags=0, time=0)
                next_ind += 1
            ret = tables.nodes.add_row(
                flags=tskit.NODE_IS_SAMPLE if ind.is_alive is True else 0,
                time=self.time - ind.time,
            )
            assert ret == ind.index
            next_ind += 1

        for ind in sorted_individuals:
            for child, segments in ind.children.items():
                for seg in segments:
                    tables.edges.add_row(
                        left=seg.left,
                        right=seg.right,
                        parent=ind.index,
                        child=child.index,
                    )
        # Can't be bothered doing the sorting above to get rid of this,
        # but it's trivial.
        tables.sort()
        return tables.tree_sequence()


def main():
    seed = 2123501251
    # sim = Simulator(100, 5, death_proba=1.0, seed=seed)
    # sim = Simulator(6, 5, death_proba=1.0, seed=seed)
    sim = Simulator(8, 5, death_proba=0.5, seed=seed)
    # works for 1 generation...
    # sim.run(1)
    sim.run(200)
    ts = sim.export()
    print(ts.draw_text())

    tables, samples, node_map = sim.convert_transmissions_to_tables()
    # print(samples)
    idmap = tables.simplify(samples)
    ts_tsk = tables.tree_sequence()

    node_map = {v: k for k, v in node_map.items()}

    # Make node labels which should match what we
    # draw from the new algo.
    node_labels = {}
    for i, j in enumerate(idmap):
        if j != tskit.NULL:
            node_labels[j] = str(node_map[i])
    print(ts_tsk.draw_text(node_labels=node_labels))


if __name__ == "__main__":
    main()


def test_remove_self_mapping():
    L = 5
    child = Individual(0, 0, is_alive=True)
    child.ancestry = [Segment(0, L, child)]
    assert len(child.ancestry) == 1
    child.remove_sample_mapping(L)
    assert len(child.ancestry) == 0


def test_basics():
    """
    Simple case, easy to reason through.
    """
    pop = []
    L = 5
    for i in range(2):
        parent = Individual(0, i, is_alive=False)
        assert parent.index == i
        pop.append(parent)

    # transmission w/coalescence
    c = Individual(1, True)
    c.ancestry.append(Segment(0, L, c))
    record_inheritance(0, L // 2, pop[0], c)
    cc = Individual(1, True)
    cc.ancestry.append(Segment(0, L, cc))
    record_inheritance(L // 3, 3 * L // 4, pop[0], cc)

    # 2x unary edges
    record_inheritance(0, L // 3, pop[1], cc)
    record_inheritance(3 * L // 4, L, pop[1], cc)

    assert pop[0] in c.parents
    assert pop[1] not in c.parents
    assert pop[0] in cc.parents
    assert pop[1] in cc.parents

    e = pop[0].intersecting_ancestry()
    assert len(e) == 2

    propagate_upwards(pop[0])

    assert len(pop[0].ancestry) == 3
    assert Segment(0, 1, c) in pop[0].ancestry
    assert Segment(1, 2, pop[0]) in pop[0].ancestry
    assert Segment(2, 3, cc) in pop[0].ancestry

    propagate_upwards(pop[1])
    assert len(pop[1].ancestry) == 2
    assert Segment(0, L // 3, cc) in pop[1].ancestry
    assert Segment(3 * L // 4, L, cc) in pop[1].ancestry

    propagate_upwards(c)
    propagate_upwards(cc)

    assert len(pop[0].ancestry) == 3
    assert Segment(0, 1, c) in pop[0].ancestry
    assert Segment(1, 2, pop[0]) in pop[0].ancestry
    assert Segment(2, 3, cc) in pop[0].ancestry

    assert len(pop[1].ancestry) == 2
    assert Segment(0, L // 3, cc) in pop[1].ancestry
    assert Segment(3 * L // 4, L, cc) in pop[1].ancestry


def failing_case_1():
    # seed = 1, N=4, L=5, using basically Jerome's prototype
    pop = []
    L = 5
    for i in range(4):
        parent = Individual(0, i, is_alive=False)
        assert parent.index == i
        pop.append(parent)

    next_index = len(pop)

    replacements = []
    x = [1, 4, 1, 1]  # xover positions
    parents = [(0, 2), (3, 3), (0, 3), (3, 3)]
    for i in range(4):
        child = Individual(1, next_index)
        next_index += 1
        child.ancestry.append(Segment(0, L, child))
        record_inheritance(0, x[i], pop[parents[i][0]], child)
        record_inheritance(x[i], L, pop[parents[i][1]], child)
        replacements.append((i, child))
    for j, ind in replacements:
        dead = pop[j]
        dead.is_alive = False
        # NOTE: EXPERIMENTAL
        dead.remove_sample_mapping(sequence_length=L)
        propagate_upwards(dead)
        pop[j] = ind

    for _, ind in replacements:
        # print("replacement")
        # ind.print_state()
        propagate_upwards(ind)

    return pop


def failing_case_2():
    # seed = 2, N=4, L=5, using basically Jerome's prototype
    pop = []
    L = 5
    for i in range(4):
        parent = Individual(0, i, is_alive=False)
        assert parent.index == i
        pop.append(parent)
    next_index = len(pop)

    replacements = []
    x = [1, 3, 4, 3]  # xover positions
    parents = [(0, 0), (1, 2), (0, 1), (2, 3)]
    for i in range(4):
        child = Individual(1, next_index)
        next_index += 1
        child.ancestry.append(Segment(0, L, child))
        record_inheritance(0, x[i], pop[parents[i][0]], child)
        record_inheritance(x[i], L, pop[parents[i][1]], child)
        replacements.append((i, child))
    for j, ind in replacements:
        dead = pop[j]
        dead.is_alive = False
        # NOTE: EXPERIMENTAL
        dead.remove_sample_mapping(sequence_length=L)
        propagate_upwards(dead)
        pop[j] = ind

    for _, ind in replacements:
        # print("replacement")
        # ind.print_state()
        propagate_upwards(ind)

    return pop


def collect_unique_individuals(pop):
    individuals = set()
    for i in pop:
        stack = [i]
        while len(stack) > 0:
            j = stack.pop()
            individuals.add(j)
            for p in j.parents:
                stack.append(p)
    return individuals


def test_failing_case_1():
    pop = failing_case_1()
    individuals = collect_unique_individuals(pop)

    parent_indexes = [(), (), (), (), (0,), (3,), (3, 0), (3,)]

    for i in individuals:
        assert len(i.parents) == len(parent_indexes[i.index]), f"{i} -> {i.parents}"
        for p in i.parents:
            assert p.index in parent_indexes[i.index]

    for i in individuals:
        if i.index > 3:
            assert i.is_alive
            assert len(i.ancestry) == 1
        else:
            assert not i.is_alive

            assert i.index != 1
            assert i.index != 2

            if i.index == 0:
                assert len(i.ancestry) == 1
                assert i.ancestry[0].child is i
                assert i.ancestry[0].left == 0
                assert i.ancestry[0].right == 1

                found4 = False
                found6 = False
                for child, segments in i.children.items():
                    if child.index == 4:
                        assert Segment(0, 1, None) in segments
                        found4 = True
                    elif child.index == 6:
                        assert Segment(0, 1, None) in segments
                        found6 = True
                    elif child.index == i.index:
                        pass
                    else:
                        assert False, f"{child}"

                assert found4 is True
                assert found6 is True

            if i.index == 3:
                i.print_state()
                assert len(i.ancestry) == 3
                segs = [(0, 1), (1, 4), (4, 5)]
                for a, s in zip(i.ancestry, segs):
                    assert a.left == s[0]
                    assert a.right == s[1]
                    assert a.child is i

                found5 = False
                found6 = False
                found7 = False
                for child, segments in i.children.items():
                    if child.index == 5:
                        assert Segment(0, 1, None) in segments
                        assert Segment(1, 4, None) in segments
                        assert Segment(4, 5, None) in segments
                        found5 = True
                    elif child.index == 7:
                        assert Segment(0, 1, None) in segments
                        assert Segment(1, 4, None) in segments
                        assert Segment(4, 5, None) in segments
                        found7 = True
                    elif child.index == 6:
                        assert Segment(1, 4, None) in segments
                        assert Segment(4, 5, None) in segments
                        found6 = True
                    elif child.index == i.index:
                        pass
                    else:
                        pass
                        assert False, f"{child}"

                assert found5 is True, "A"
                assert found6 is True, "B"
                assert found7 is True, "C"


def test_failing_case_2():
    pop = failing_case_2()
    individuals = collect_unique_individuals(pop)

    parent_indexes = [(), (), (), (), (0,), (), (0,), ()]

    # for i in individuals:
    #     if i.index == 0:
    #         i.print_state()
    #     print("CHILDREN")
    #     for j in i.children:
    #         j.print_state()

    for i in individuals:
        assert len(i.parents) == len(parent_indexes[i.index]), f"{i} -> {i.parents}"
        for p in i.parents:
            assert p.index in parent_indexes[i.index]


def test_failing_case_2_subtree():
    """
    The "problem" involves node 0 and its 2 offspring.

    This test only records inheritances from 0 to the 2 offspring.
    """
    pop = []
    L = 5
    for i in range(4):
        parent = Individual(0, i, is_alive=False)
        assert parent.index == i
        pop.append(parent)

    next_index = len(pop)

    replacements = []
    x = [1, 3, 4, 3]  # xover positions
    parents = [(0, 0), (1, 2), (0, 1), (2, 3)]
    for i in range(4):
        child = Individual(1, next_index)
        next_index += 1
        child.ancestry.append(Segment(0, L, child))
        p1 = False
        p2 = False
        if parents[i][0] == 0 or parents[i][0] == 1:
            p1 = True
        if parents[i][1] == 0 or parents[i][1] == 1:
            p2 = True
        if parents[i][0] == 0 or parents[i][1] == 0:
            record_inheritance(0, x[i], pop[parents[i][0]], child)
            record_inheritance(x[i], L, pop[parents[i][1]], child)
        replacements.append((i, child))

    print("propagate deaths")
    for j, ind in replacements:
        dead = pop[j]
        dead.is_alive = False
        # NOTE: EXPERIMENTAL
        dead.remove_sample_mapping(sequence_length=L)
        propagate_upwards(dead)
        pop[j] = ind

    print("propagate replacements")
    for _, ind in replacements:
        # print("replacement")
        # ind.print_state()
        propagate_upwards(ind)

    individuals = collect_unique_individuals(pop)

    parent_indexes = [(), (), (), (), (0,), (), (0,), ()]

    for i in individuals:
        assert len(i.parents) == len(parent_indexes[i.index]), f"{i} -> {i.parents}"
        for p in i.parents:
            assert p.index in parent_indexes[i.index]


def test_failing_case_1_next_generation():
    L = 5
    pop = failing_case_1()

    next_index = max([i.index for i in collect_unique_individuals(pop)]) + 1

    replacements = []
    x = [1, 1, 2, 4]  # xover positions
    parents = [(2, 1), (0, 0), (0, 3), (0, 1)]
    for i in range(4):
        child = Individual(2, next_index)
        next_index += 1
        child.ancestry.append(Segment(0, L, child))
        record_inheritance(0, x[i], pop[parents[i][0]], child)
        record_inheritance(x[i], L, pop[parents[i][1]], child)
        replacements.append((i, child))
    for j, ind in replacements:
        dead = pop[j]
        dead.is_alive = False
        # NOTE: EXPERIMENTAL
        dead.remove_sample_mapping(sequence_length=L)
        propagate_upwards(dead)
        pop[j] = ind

    for _, ind in replacements:
        # print("replacement")
        # ind.print_state()
        propagate_upwards(ind)
