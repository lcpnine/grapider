"""Deterministic DAG store and validators. No LLM anywhere in this module."""
import json

import networkx as nx

MAX_PARENTS = 3


class PrereqDAG:
    def __init__(self):
        self.g = nx.DiGraph()
        self.retired = {}

    def add_node(self, node_id, name=None, provenance=(), valid_versions=()):
        """Add a node, or merge provenance and versions into an existing one.

        A retired concept that comes back keeps its version history, but not its old edges.
        """
        if node_id not in self.g:
            prior = self.retired.pop(node_id, {})
            self.g.add_node(node_id, name=name or prior.get("name") or node_id, provenance=set(),
                            valid_versions=set(prior.get("valid_versions", ())))
        attrs = self.g.nodes[node_id]
        attrs["provenance"] |= set(provenance)
        attrs["valid_versions"] |= set(valid_versions)

    def add_edge(self, prereq, dependent, reason):
        """Add prereq -> dependent. Cycles are allowed here and reported by validate()."""
        reason = reason.strip()
        if not reason or "\n" in reason:
            raise ValueError("an edge needs a one-line reason")
        for node in (prereq, dependent):
            if node not in self.g:
                raise KeyError(f"unknown node {node}")
        if prereq == dependent:
            raise ValueError(f"self-loop on {prereq}")
        self.g.add_edge(prereq, dependent, reason=reason)

    def retire(self, node_id, version):
        """Take a node out of the active graph, and keep a record of what it was."""
        attrs = self.g.nodes[node_id]
        self.retired[node_id] = dict(
            name=attrs["name"], provenance=sorted(attrs["provenance"]),
            valid_versions=sorted(attrs["valid_versions"]), retired_in=version,
            parents=sorted(self.g.predecessors(node_id)), children=sorted(self.g.successors(node_id)),
        )
        self.g.remove_node(node_id)

    def validate(self):
        """Return {check: violations}. Every check is present; empty means it passed."""
        cycles = list(nx.simple_cycles(self.g))
        return {
            "cycles": cycles,
            "too_many_parents": sorted(n for n in self.g if self.g.in_degree(n) > MAX_PARENTS),
            "redundant_edges": [] if cycles else sorted(self._redundant_edges()),
            "orphans": sorted(n for n in self.g if self.g.degree(n) == 0) if len(self.g) > 1 else [],
        }

    def _redundant_edges(self):
        reduced = nx.transitive_reduction(self.g)
        return set(self.g.edges) - set(reduced.edges)

    def transitive_reduce(self):
        """Remove redundant edges and return them. Reachability is unchanged."""
        if not nx.is_directed_acyclic_graph(self.g):
            raise ValueError("cannot reduce a cyclic graph")
        removed = sorted(self._redundant_edges())
        self.g.remove_edges_from(removed)
        return removed

    def path_to(self, target, known=()):
        """Stable topological study order for target, skipping concepts already known."""
        needed = (nx.ancestors(self.g, target) | {target}) - set(known)
        return list(nx.lexicographical_topological_sort(self.g.subgraph(needed)))

    def closure(self):
        """All (prereq, dependent) pairs connected by a path."""
        return set(nx.transitive_closure_dag(self.g).edges)

    def to_json(self):
        return dict(
            nodes=[
                dict(id=n, name=a["name"], provenance=sorted(a["provenance"]),
                     valid_versions=sorted(a["valid_versions"]))
                for n, a in sorted(self.g.nodes(data=True))
            ],
            edges=[dict(prereq=u, dependent=v, reason=a["reason"])
                   for u, v, a in sorted(self.g.edges(data=True))],
            retired=self.retired,
        )

    @classmethod
    def from_json(cls, data):
        dag = cls()
        for n in data["nodes"]:
            dag.add_node(n["id"], n["name"], n["provenance"], n["valid_versions"])
        for e in data["edges"]:
            dag.add_edge(e["prereq"], e["dependent"], e["reason"])
        dag.retired = json.loads(json.dumps(data.get("retired", {})))
        return dag

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_json(json.load(f))
