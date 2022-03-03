"""
Prototype forward simulator where we simplify inline, using the Individual
structures directly.
"""
import random
import collections
import heapq
import sys

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


def propagate_upwards(ind, clear_if_not_alive: bool = False):
    pass
    # This isn't working.
    # print("PROPAGATE", ind)
    stack = [ind]
    last_time = int(np.iinfo(np.uint32).max)
    while len(stack) > 0:
        ind = stack.pop()
        # assert ind.time <= last_time, f"{ind.time} {last_time}"
        last_time = ind.time
        # print("\t", ind)
        # ind.print_state()
        # We're visting everything here at the moment, but we don't need to.
        # We should only have to visit the parents for which we have ancestral
        # segents, and so the areas of the graph we traverse should be
        # quickly localised.
        # print("before")
        # ind.print_state()
        # print(f"updating {ind}")
        ind.update_ancestry(clear_if_not_alive)
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

    # This is just for debugging
    _next_index = 0

    def __init__(self, time, is_alive=True):
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
        self.parents = set()
        self.is_alive = is_alive
        self.index = Individual._next_index
        Individual._next_index += 1

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

    def update_ancestry(self, processing_replacements: bool = False):
        # FIXME: (??) -- the second time we simplify,
        # the existing ancestry is already "full"...
        S = self.intersecting_ancestry()
        # for child in self.children.keys():
        #     child.parents.remove(self)
        self.children.clear()

        # NOTE: this is a trivial, but undesirable
        # fix to the problem of adding redudant ancestry
        # when propagating upwards from replacements
        # if not self.is_alive:
        #     self.ancestry.clear()

        ancestry_i = 0

        for left, right, X in overlapping_segments(S):
            # print(left, right, X)
            if len(X) == 1:
                # print("unary")
                mapped_ind = X[0].child
                print(self, "unary", left, right, mapped_ind)
                if self in mapped_ind.parents:
                    if self.index == 0:
                        print(self.ancestry)

                    # This is only correct for THIS SEGMENT
                    # but is wrong in general.
                    # The "failing_case_2" is an example
                    # of where this is removed, but there
                    # are other, non-overlapping segs existing as edges
                    # mapped_ind.parents.remove(self)

                    # Remove the parent i.f.f there
                    # are no other edges mapping to this same child
                    non_overlapping_edges_exist = False
                    if mapped_ind in self.children:
                        for seg in self.children[mapped_ind]:
                            if not (seg.right > X[0].left and X[0].right > seg.left):
                                non_overlapping_edges_exist = True
                                break

                    if not non_overlapping_edges_exist:
                        print(
                            f"Removing parent {self.index} from child {mapped_ind.index}"
                        )
                        mapped_ind.parents.remove(self)
                    else:
                        assert self in mapped_ind.parents

            else:
                mapped_ind = self
                for x in X:
                    print("adding edge to", self, ":", x.child, left, right)
                    print(f"parents of {x.child} -> {x.child.parents}")
                    self.add_child_segment(x.child, left, right)
                    # assert self in x.child.parents
                    # NOTE: the "need" for this may be a side
                    # effect of incorrect logic in the propagation
                    # steps.
                    # NOTE: this is actually WRONG in general,
                    # as it leads to long-term retention of unary nodes.
                    # x.child.parents.add(self)
                assert_non_overlapping(self.children[mapped_ind])
            # If an individual is alive it always has ancestry over the
            # full segment, so we don't overwrite this.
            if not self.is_alive:
                seg = Segment(left, right, mapped_ind)
                print(f"Adding ancestry {seg} to {self.ancestry} of {self.index}")
                new_segment = True
                for i in self.ancestry:
                    if i.right > seg.left and seg.right > i.left:
                        new_segment = False
                if new_segment:
                    self.ancestry.append(seg)
                # TODO: figure out how/why we are trying
                # to put redundant segs into self.ancestry.
                # NOTE: what is happening here is that
                # we are constantly trying to remap nodes onto self.
                # NOTE: this is happening during ancestry propagation
                # for replacement individuals.
                # NOTE: there are multiple complexities here:
                # 1. Fundamentally, we have to deal with already-filled
                #    self.ancestry.
                # 2. To do so, we (probably?) need to be co-iterating
                #    over the overlaps (X) and self.ancestry,  updating
                #    the contents of the latter such that:
                #    a. unary transmissions "upgrade" to coalescences.
                #    b. changes in [left, right) mapping to the
                #       same non-self child get updated.

                # if not processing_replacements:
                #    # If a segment length has changed,
                #    # find it and update it
                #    # NOTE: this may be the wrong way to go...
                #    found = False
                #    for i in self.ancestry:
                #        if i.child.index == mapped_ind.index:
                #            if i.left == seg.left:
                #                seg.left = i.left
                #                found = True
                #                break
                #            elif i.right == seg.right:
                #                seg.left = i.left
                #                found = True
                #                break
                #    if not found:
                #        print(f"ADDING {seg} to ancestry of {self}")
                #        self.ancestry.append(seg)
                # elif mapped_ind.index != self.index:
                #     self.ancestry.append(seg)

                # found_seg = None

                # TODO: this is not correct.
                # We need to be looking explicitly for overlaps
                # here and dealing with them as they come up.
                # while (
                #     ancestry_i < len(self.ancestry)
                #     and self.ancestry[ancestry_i].left <= seg.left
                # ):
                #     found_seg = self.ancestry[ancestry_i]
                #     ancestry_i += 1

                # # print(f"found seg = {found_seg} {seg}")

                # if found_seg is not None:
                #     if found_seg.left == seg.left:
                #         if seg.child.index == self.index:
                #             # print(f"maps to self: {self}")
                #             # overlaps a known coalescent
                #             pass
                #         else:
                #             # overlaps an existing ancestry seg
                #             if seg.child.index == found_seg.child.index:
                #                 found_seg.left = min(seg.left, found_seg.left)
                #             else:
                #                 print(f"promoting to coalsecence {found_seg} {seg}")
                #                 found_seg.left = max(found_seg.left, seg.left)
                #                 found_seg.right = min(found_seg.right, seg.right)
                #                 # non-coalescence promoted to coalescence
                #                 found_seg.child = self
                #                 print(f"promoted to coalsecence {found_seg} {seg}")
                # else:
                #     # print(f"ancestry_i = {ancestry_i}")
                #     if ancestry_i < len(self.ancestry):
                #         self.ancestry.insert(ancestry_i, seg)
                #     else:
                #         self.ancestry.append(seg)

                # print("pre-assert")
                # self.print_state()
                assert_non_overlapping(self.ancestry)
                # if seg not in self.ancestry:
                #     self.ancestry.append(seg)
                # else:
                #     print("redundant")
                # NOTE: something like below
                # is almost certainly wrong for large genomic
                # lengths and infrequent simplification
                # if seg.child is not self:
                #     self.ancestry.append(seg)


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
        self.population = [Individual(self.time) for _ in range(population_size)]

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
                left_parent_index = -1
                for k in range(self.population_size):
                    if left_parent is self.population[k]:
                        left_parent_index = k
                        break
                right_parent_index = -1
                for k in range(self.population_size):
                    if right_parent is self.population[k]:
                        right_parent_index = k
                        break
                # Using integers here just to make debugging easier
                x = self.rng.randint(1, self.sequence_length - 1)
                assert 0 < x < self.sequence_length
                print(
                    "BIRTH:",
                    left_parent,
                    left_parent_index,
                    right_parent,
                    right_parent_index,
                    j,
                    x,
                )
                child = Individual(self.time)
                child.ancestry = [Segment(0, self.sequence_length, child)]
                replacements.append((j, child))
                record_inheritance(0, x, left_parent, child)
                record_inheritance(x, self.sequence_length, right_parent, child)

        # First propagate the loss of the ancestral material from the newly dead
        print("pdead")
        for j, ind in replacements:
            # print("replacement")
            # ind.print_state()
            dead = self.population[j]
            # NOTE: here, we have a few problems, as far as simulating > 1 generation
            # is concerned:
            # 1. If the Individual has an ancestral mapping to self on [0, L),
            #    that mapping is no longer valid.
            # 2. We have ancestry already in place for  the next "round" of evolution.
            #    So we are now in a position where we may are not BUILDING ancestry
            #    de novo, but rather updating ancestry that is already in memory.
            dead.is_alive = False
            # NOTE: EXPERIMENTAL
            dead.remove_sample_mapping(sequence_length=self.sequence_length)
            propagate_upwards(dead)
            self.population[j] = ind
        # print("done")
        # Now propagate the gain in the ancestral material from the children upwards.
        print("prepl")
        for _, ind in replacements:
            # print("replacement")
            # ind.print_state()
            propagate_upwards(ind, True)
        self.check_state()

        # for ind in self.all_reachable():
        #     ind.print_state()

    def check_state(self):
        for ind in self.all_reachable():
            print(f"REACHABLE: {ind.index}")
            ind.print_state()
            # print("reachable individiual:")
            # ind.print_state()
            if ind.is_alive:
                assert len(ind.ancestry) == 1
                x = ind.ancestry[0]
                assert x.left == 0
                assert x.right == self.sequence_length
                assert x.child == ind
            else:
                assert_non_overlapping(ind.ancestry)
            for child, segments in ind.children.items():
                assert_non_overlapping(segments)
                # FIXME: WHY/HOW are we getting self added
                # as child of self?
                if child is not ind:
                    assert ind in child.parents, f"{ind} {child}"
            for parent in ind.parents:
                if ind not in parent.children:
                    print("the failing parent is")
                    parent.print_state()
                    print("done w/failing parent")
                assert ind in parent.children

    def run(self, num_generations, simplify_interval=1):
        for _ in range(num_generations):
            # NOTE: this does not help...
            # for i in self.population:
            #     if not i.is_alive:
            #         i.ancestry.clear()
            self.run_generation()
            # print(len(self.dead))

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
                        print(
                            f"REACHING: parent {parent.index} is reachable via {ind.index}"
                        )
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
    seed = 1
    # sim = Simulator(100, 5, death_proba=1.0, seed=seed)
    sim = Simulator(4, 5, death_proba=1.0, seed=seed)
    # works for 1 generation...
    # sim.run(1)
    sim.run(2)
    ts = sim.export()
    print(ts.draw_text())

    stack = [sim.population[0]]
    while len(stack) > 0:
        ind = stack.pop()
        if ind.index == 21:
            ind.print_state()
            print("the parents of 21")
            for p in ind.parents:
                p.print_state()
            break
        for p in ind.parents:
            stack.append(p)
    # ts_simplify = ts.simplify()
    # print(ts_simplify.draw_text())


if __name__ == "__main__":
    main()


def test_remove_self_mapping():
    L = 5
    Individual._next_index = 0
    child = Individual(0, is_alive=True)
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
    Individual._next_index = 0
    for i in range(2):
        parent = Individual(0, is_alive=False)
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

    propagate_upwards(pop[0], False)

    assert len(pop[0].ancestry) == 3
    assert Segment(0, 1, c) in pop[0].ancestry
    assert Segment(1, 2, pop[0]) in pop[0].ancestry
    assert Segment(2, 3, cc) in pop[0].ancestry

    propagate_upwards(pop[1], False)
    assert len(pop[1].ancestry) == 2
    assert Segment(0, L // 3, cc) in pop[1].ancestry
    assert Segment(3 * L // 4, L, cc) in pop[1].ancestry

    propagate_upwards(c, True)
    propagate_upwards(cc, True)

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
    Individual._next_index = 0
    for i in range(4):
        parent = Individual(0, is_alive=False)
        assert parent.index == i
        pop.append(parent)

    replacements = []
    x = [1, 4, 1, 1]  # xover positions
    parents = [(0, 2), (3, 3), (0, 3), (3, 3)]
    for i in range(4):
        child = Individual(1)
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
        propagate_upwards(ind, True)

    return pop


def failing_case_2():
    # seed = 2, N=4, L=5, using basically Jerome's prototype
    pop = []
    L = 5
    Individual._next_index = 0
    for i in range(4):
        parent = Individual(0, is_alive=False)
        assert parent.index == i
        pop.append(parent)

    replacements = []
    x = [1, 3, 4, 3]  # xover positions
    parents = [(0, 0), (1, 2), (0, 1), (2, 3)]
    for i in range(4):
        child = Individual(1)
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
        propagate_upwards(ind, True)

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

            if i.index == 3:
                assert len(i.ancestry) == 3
                segs = [(0, 1), (1, 4), (4, 5)]
                for a, s in zip(i.ancestry, segs):
                    assert a.left == s[0]
                    assert a.right == s[1]
                    assert a.child is i


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
    Individual._next_index = 0
    for i in range(4):
        parent = Individual(0, is_alive=False)
        assert parent.index == i
        pop.append(parent)

    replacements = []
    x = [1, 3, 4, 3]  # xover positions
    parents = [(0, 0), (1, 2), (0, 1), (2, 3)]
    for i in range(4):
        child = Individual(1)
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
        propagate_upwards(ind, True)

    individuals = collect_unique_individuals(pop)

    parent_indexes = [(), (), (), (), (0,), (), (0,), ()]

    for i in individuals:
        assert len(i.parents) == len(parent_indexes[i.index]), f"{i} -> {i.parents}"
        for p in i.parents:
            assert p.index in parent_indexes[i.index]


def test_failing_case_1_next_generation():
    L = 5
    pop = failing_case_1()

    replacements = []
    x = [1, 1, 2, 4]  # xover positions
    parents = [(2, 1), (0, 0), (0, 3), (0, 1)]
    for i in range(4):
        child = Individual(2)
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
        propagate_upwards(ind, True)
