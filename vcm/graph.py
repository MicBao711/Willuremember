"""Helpers for the dependency DAG (edges point from a unit to what it is built from)."""

from __future__ import annotations

from typing import Iterable, Mapping


def find_cycle(edges: Mapping[str, Iterable[str]]) -> list[str] | None:
    """Return one cycle as a node list, or None if the graph is acyclic."""
    WHITE, GREY, BLACK = 0, 1, 2
    colour: dict[str, int] = {}
    parent: dict[str, str] = {}

    for root in edges:
        if colour.get(root, WHITE) != WHITE:
            continue
        stack = [(root, iter(edges.get(root, ())))]
        colour[root] = GREY
        while stack:
            node, children = stack[-1]
            child = next(children, None)
            if child is None:
                colour[node] = BLACK
                stack.pop()
                continue
            state = colour.get(child, WHITE)
            if state == GREY:
                cycle = [child, node]
                while cycle[-1] != child:
                    cycle.append(parent[cycle[-1]])
                return cycle[::-1]
            if state == WHITE:
                parent[child] = node
                colour[child] = GREY
                stack.append((child, iter(edges.get(child, ()))))
    return None


def topological_order(edges: Mapping[str, Iterable[str]]) -> list[str]:
    """Dependencies first. Raises ValueError on a cycle."""
    cycle = find_cycle(edges)
    if cycle:
        raise ValueError(f"dependency cycle: {' -> '.join(cycle)}")
    order: list[str] = []
    seen: set[str] = set()

    def visit(n: str) -> None:
        if n in seen:
            return
        seen.add(n)
        for d in edges.get(n, ()):
            visit(d)
        order.append(n)

    for n in edges:
        visit(n)
    return order


def dependents(edges: Mapping[str, Iterable[str]], target: str) -> set[str]:
    """All nodes that (transitively) depend on `target`."""
    reverse: dict[str, set[str]] = {}
    for n, ds in edges.items():
        for d in ds:
            reverse.setdefault(d, set()).add(n)
    out: set[str] = set()
    frontier = [target]
    while frontier:
        for n in reverse.get(frontier.pop(), ()):
            if n not in out:
                out.add(n)
                frontier.append(n)
    return out
