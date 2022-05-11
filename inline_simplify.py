"""
Prototype forward simulator where we simplify inline, using the Individual
structures directly.
"""
import argparse
from dataclasses import dataclass
from enum import IntEnum
import random
import collections
import heapq
import sys
from typing import List, Tuple, Dict, Optional

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


def make_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-N", type=int, default=8, help="Haploid population size")
    parser.add_argument(
        "--death_probability", "-d", type=float, default=0.5, help="Death probability"
    )
    parser.add_argument(
        "--simlen",
        "-s",
        type=int,
        default=200,
        help="Simulation length (number of birth steps)",
    )
    parser.add_argument(
        "--genome_length", "-L", type=int, default=5, help="Genome length"
    )
    parser.add_argument(
        "--seed", "-S", type=int, default=501251, help="Random number seed"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Print excessive amount of stuff to the screen",
    )

    return parser


def validate_args(args):
    assert args.N > 1
    assert args.seed >= 0
    assert args.simlen > 0
    assert args.death_probability > 0.0 and args.death_probability <= 1.0
    assert args.genome_length > 1


@dataclass
class ParentNode:
    time: float
    children: List[int]

    def __post_init__(self):
        self.children = sorted(self.children)


@dataclass
class Topology:
    left: int
    right: int
    parents: Dict[int, ParentNode]


def make_topologies(ts: tskit.TreeSequence, node_labels=None):
    topologies = []

    for t in ts.trees():
        topo = {}
        for n in t.preorder():
            if node_labels is None:
                topo[n] = ParentNode(ts.node(n).time, [c for c in t.children(n)])
            else:
                # Back-label from tkit's IDs to those used in this prototype
                p = node_labels[n]
                topo[p] = ParentNode(
                    ts.node(n).time, [node_labels[c] for c in t.children(n)]
                )
        temp = Topology(int(t.interval[0]), int(t.interval[1]), topo)
        if len(topologies) > 0:
            # Deal w/the fact that we are not squashing edges.
            if temp.parents != topologies[-1].parents:
                topologies.append(temp)
            else:
                topologies[-1].right = temp.right
        else:
            topologies.append(temp)

    return topologies


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


def propagate_upwards(individuals, verbose, sequence_length):
    heapq.heapify(individuals)
    current_ind = set([i.individual for i in individuals])
    last_time = None
    processed = set()

    visits = 0

    while len(individuals) > 0:
        assert len(individuals) == len(current_ind)
        ind = heapq.heappop(individuals)
        visits += 1
        current_ind.remove(ind.individual)
        if last_time is not None:
            assert (
                ind.individual.time <= last_time
            ), f"{ind.individual.time} {last_time}"
            last_time = ind.individual.time
        else:
            last_time = ind.individual.time
        assert ind.individual not in processed, f"{ind.individual}"
        processed.add(ind.individual)

        if ind.type == IndividualType.DEATH:
            ind.individual.is_alive = False
            ind.individual.remove_sample_mapping(sequence_length=sequence_length)

        changed = ind.individual.update_ancestry(verbose)
        if changed or ind.individual.is_alive:
            for parent in ind.individual.parents:
                if parent not in current_ind:
                    assert parent.time < ind.individual.time
                    if last_time is not None:
                        assert parent.time < last_time, f"{parent.time}, {last_time}"
                    heapq.heappush(
                        individuals, IndividualToProcess(parent, IndividualType.PARENT)
                    )
                    current_ind.add(parent)
    return visits


def record_inheritance(left, right, parent, child):
    """
    Record the inheritances of genetic material from parent to child over the
    from coordinate left to right.
    """
    child.parents.add(parent)
    parent.add_child_segment(child, left, right)


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


class AncestryOverlap(object):
    """
    An ancestral segment mapping a given individual to a half-open genomic
    interval [left, right).
    """

    def __init__(self, left, right, child, mapped_node):
        self.left = left
        self.right = right
        self.child = child
        self.mapped_node = mapped_node

    def __repr__(self):
        return repr((self.left, self.right, self.child, self.mapped_node))

    def __eq__(self, o):
        return (
            self.left == o.left
            and self.right == o.right
            and self.child == o.child
            and self.mapped_node == o.mapped_node
        )


@dataclass
class ChildInputDetails:
    input_number_segs: int
    output_number_segs: int


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

    def __hash__(self) -> int:
        return self.index

    def __lt__(self, other):
        """
        Sort by birth time present -> past
        """
        return self.time > other.time

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

    def update_child_segments(
        self,
        child,
        left,
        right,
    ):
        segs = self.children[child]
        if len(segs) > 0:
            if segs[-1].right == left:
                segs[-1].right = right
            else:
                seg = Segment(left, right, None)
                segs.append(seg)
        else:
            seg = Segment(left, right, None)
            segs.append(seg)

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
                        y = AncestryOverlap(
                            max(x.left, e.left), min(x.right, e.right), child, x.child
                        )
                        S.append(y)
        return S

    def update_ancestry(self, verbose):
        S = self.intersecting_ancestry()

        if verbose is True:
            print(f"START {self.index}")
            print(f"S = {S}")
            for c in self.children.keys():
                print(f"child {c} has ancestry {c.ancestry}, children {c.children}")
            self.print_state()

        input_ancestry_len = len(self.ancestry)
        output_ancestry_index = 0

        ancestry_change_detected = False

        for k in self.children.keys():
            k.parents.remove(self)

        self.children.clear()

        for left, right, X in overlapping_segments(S):
            assert right > left
            if verbose is True:
                print("ANCESTRY info = ", left, right, X)
            if len(X) == 1:
                mapped_ind = X[0].mapped_node
                if verbose is True:
                    print("unary", mapped_ind, X[0])
                if self.is_alive:
                    self.update_child_segments(
                        mapped_ind,
                        left,
                        right,
                    )
            else:
                mapped_ind = self

                for x in X:
                    self.update_child_segments(
                        x.mapped_node,
                        left,
                        right,
                    )

                if mapped_ind in self.children:
                    # NOTE: this is a really annoyting gotcha:
                    # the defaultdict(list) adds [] to self.children
                    # if it does not exist
                    assert_non_overlapping(self.children[mapped_ind])
            # If an individual is alive it always has ancestry over the
            # full segment, so we don't overwrite this.
            if not self.is_alive:
                seg = Segment(left, right, mapped_ind)
                if output_ancestry_index < input_ancestry_len:
                    if self.ancestry[output_ancestry_index] != seg:
                        self.ancestry[output_ancestry_index] = seg
                        ancestry_change_detected = True
                else:
                    ancestry_change_detected = True
                    self.ancestry.append(seg)
                output_ancestry_index += 1

        if not self.is_alive:
            if output_ancestry_index < input_ancestry_len:
                ancestry_change_detected = True
                del self.ancestry[output_ancestry_index:]

        assert_non_overlapping(self.ancestry)

        for c in self.children.keys():
            c.parents.add(self)

        for a in self.ancestry:
            if a.child in self.children and a.child is not self:
                assert self in a.child.parents, f"{self} {a} {self.ancestry}"
        if verbose is True:
            self.print_state()
            print("DONE")

        assert self not in self.parents
        # Something CLOSE TO, but not EXACTLY, like the
        # next line gets on on the path to not visiting
        # the entire graph.
        rv = ancestry_change_detected or len(self.ancestry) == 0
        return rv


class IndividualType(IntEnum):
    PARENT = 0
    BIRTH = 1
    DEATH = 2


class IndividualToProcess(object):
    def __init__(self, individual: Individual, type: IndividualType):
        self.individual = individual
        self.type = type

    def __lt__(self, other):
        return (self.individual.time, self.type) > (other.individual.time, other.type)

    def __repr__(self):
        return f"({self.individual}, type={self.type})"


@dataclass
class TransmissionInfo:
    """
    Used for export to tskit
    """

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

    def run_generation(self, verbose):
        """
        Implements a single generation.
        """
        self.time += 1
        replacements = []
        for j in range(self.population_size):
            if self.rng.random() < self.death_proba:
                left_parent = self.rng.choice(self.population)
                right_parent = self.rng.choice(self.population)
                x = self.rng.randint(1, self.sequence_length - 1)
                assert 0 < x < self.sequence_length
                child = Individual(self.time, self.next_individual_index)
                self.next_individual_index += 1
                child.ancestry = [Segment(0, self.sequence_length, child)]
                replacements.append((j, child))
                record_inheritance(0, x, left_parent, child)
                record_inheritance(x, self.sequence_length, right_parent, child)
                assert len(child.parents) <= 2, child.parents
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

        processed = set()
        to_process = []
        for j, ind in replacements:
            dead = self.population[j]
            to_process.append(IndividualToProcess(dead, IndividualType.DEATH))
            self.population[j] = ind
        processed = set()
        nrepeats_per_birth = 0
        for _, ind in replacements:
            to_process.append(IndividualToProcess(ind, IndividualType.BIRTH))

        visits = propagate_upwards(to_process, verbose, self.sequence_length)
        print(visits)
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
                for a in ind.ancestry:
                    if a.child is not ind and a.child not in reachable:
                        if a.child.is_alive or len(a.child.children) > 0:
                            print(
                                "UNREACHABLE UNARY", ind, "->", a.child, a.child.parents
                            )
            for child, segments in ind.children.items():
                if child is not ind:
                    assert ind in child.parents
                    sys.stdout.flush()
                    assert (
                        child in reachable
                    ), f"{child} {child.parents} <-> {ind} {ind.children}"
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

    def run(self, num_generations, verbose, simplify_interval=1):
        for _ in range(num_generations):
            self.run_generation(verbose)

    def make_samples_list_for_tskit(self, node_map) -> List[int]:
        rv = []
        for i in self.population:
            if i.is_alive and node_map[i.index] not in rv:
                rv.append(node_map[i.index])
        return sorted(rv)

    def get_alive_node_indexes_and_times(self) -> List[Tuple[int, int]]:
        indexes = set()
        rv = []
        for i in self.population:
            if i.is_alive and i.index not in indexes:
                rv.append((i.index, i.time))
                indexes.add(i.index)
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
        for i in self.transmissions:
            for n in [i.parent, i.child]:
                if n.index not in node_data:
                    node_data[n.index] = n.time

        for i in self.population:
            if i.is_alive:
                node_data[i.index] = i.time

        alive_nodes = self.get_alive_node_indexes_and_times()

        assert all([a[0] in node_data for a in alive_nodes])

        node_map = {}
        for i, t in node_data.items():
            x = tables.nodes.add_row(0, -(t - self.time))
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
    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])
    validate_args(args)
    sim = Simulator(
        args.N, args.genome_length, death_proba=args.death_probability, seed=args.seed
    )
    sim.run(args.simlen, args.verbose)
    ts = sim.export()

    topologies = make_topologies(ts)

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
            node_labels[j] = node_map[i]

    tsk_topologies = make_topologies(ts_tsk, node_labels)

    for i, j in zip(topologies, tsk_topologies):
        if i != j:
            print(i)
            print(j)
            assert False

    assert topologies == tsk_topologies


if __name__ == "__main__":
    main()
