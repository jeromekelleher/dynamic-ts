"""
Prototype forward simulator where we simplify inline, using the Individual
structures directly.
"""
import random
import collections
import heapq
import sys

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
        print("parents = ", self.parents)


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

    def kill_individual(self, individual):
        """
        Kill the specified individual.
        """
        individual.is_alive = False

    def record_inheritance(self, left, right, parent, child):
        """
        Record the inheritances of genetic material from parent to child over the
        from coordinate left to right.
        """
        child.parents.add(parent)
        # TODO this should be an Interval, I guess.
        parent.children[child].append(Segment(left, right, child))

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
                # Using integers here just to make debugging easier
                x = self.rng.randint(1, self.sequence_length - 1)
                assert 0 < x < self.sequence_length
                child = Individual(self.time)
                replacements.append((j, child))
                self.record_inheritance(0, x, left_parent, child)
                self.record_inheritance(x, self.sequence_length, right_parent, child)
        for j, ind in replacements:
            self.kill_individual(self.population[j])
            self.population[j] = ind

        self.check_state()

        # for ind in self.all_reachable():
        #     ind.print_state()

    def check_state(self):
        pass

    def run(self, num_generations, simplify_interval=1):
        for _ in range(num_generations):
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
                time=self.time - ind.time)
            assert ret == ind.index
            next_ind += 1

        for ind in sorted_individuals:
            for child, segments in ind.children.items():
                for seg in segments:
                    tables.edges.add_row(
                        left=seg.left, right=seg.right,
                        parent=ind.index, child=seg.child.index)
        # Can't be bothered doing the sorting above to get rid of this,
        # but it's trivial.
        tables.sort()
        return tables.tree_sequence()


def main():
    seed = 1
    sim = Simulator(4, 5, death_proba=1.0, seed=seed)
    # sim = Simulator(400, 5, death_proba=0.5, seed=seed)
    sim.run(20)
    ts = sim.export()
    print(ts.draw_text())
    # ts_simplify = ts.simplify()
    # print(ts_simplify.draw_text())


if __name__ == "__main__":
    main()
