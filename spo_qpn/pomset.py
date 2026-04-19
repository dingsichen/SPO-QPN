"""Observation pomset canonicalization and target-prefix handling."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence


@dataclass(frozen=True)
class PomsetKey:
    labels: tuple[str, ...]
    order_pairs: tuple[tuple[int, int], ...]

    def to_display_dict(self) -> dict[str, object]:
        events = [{"id": f"e{index}", "label": label} for index, label in enumerate(self.labels)]
        order = [[f"e{src}", f"e{dst}"] for src, dst in self.order_pairs]
        topological = _one_topological_order(len(self.labels), self.order_pairs)
        return {
            "events": events,
            "order": order,
            "one_linearization": [self.labels[index] for index in topological],
        }


def canonical_pomset_key(
    visible_event_ids: Sequence[str],
    labels: Mapping[str, str],
    strict_order: Mapping[str, set[str]],
) -> PomsetKey:
    if not visible_event_ids:
        return PomsetKey(labels=(), order_pairs=())
    descendants = {event_id: set() for event_id in visible_event_ids}
    for event_id in visible_event_ids:
        for ancestor in strict_order[event_id]:
            descendants[ancestor].add(event_id)
    cover_preds, cover_succs = _cover_relations(visible_event_ids, strict_order)
    best_encoding = _canonical_search(
        tuple(sorted(visible_event_ids)),
        labels,
        strict_order,
        descendants,
        cover_preds,
        cover_succs,
        tuple(),
    )
    assert best_encoding is not None
    ordered_labels, bits = best_encoding
    order_pairs: list[tuple[int, int]] = []
    bit_pointer = 0
    for left, right in combinations(range(len(ordered_labels)), 2):
        if bits[bit_pointer] == 1:
            order_pairs.append((left, right))
        bit_pointer += 1
        if bits[bit_pointer] == 1:
            order_pairs.append((right, left))
        bit_pointer += 1
    return PomsetKey(labels=ordered_labels, order_pairs=tuple(order_pairs))


def build_target_prefix_family(target_observations: Sequence[dict[str, object]]) -> set[PomsetKey]:
    family: set[PomsetKey] = set()
    for specification in target_observations:
        family.update(_prefixes_of_spec(specification))
    return family


def build_target_family(target_observations: Sequence[dict[str, object]]) -> set[PomsetKey]:
    return {canonical_pomset_from_spec(specification) for specification in target_observations}


def canonical_pomset_from_spec(specification: dict[str, object]) -> PomsetKey:
    if "sequence" in specification:
        specification = sequence_to_spec(specification["sequence"])
    event_ids = list(specification["events"])
    labels = dict(specification["labels"])
    immediate_preds = {event_id: set() for event_id in event_ids}
    for src, dst in specification.get("order", []):
        immediate_preds[dst].add(src)
    strict_order = _transitive_closure(event_ids, immediate_preds)
    return canonical_pomset_key(event_ids, labels, strict_order)


def sequence_to_spec(sequence: Sequence[str]) -> dict[str, object]:
    event_ids = [f"v{index}" for index in range(len(sequence))]
    labels = {event_id: label for event_id, label in zip(event_ids, sequence)}
    order = [[event_ids[index], event_ids[index + 1]] for index in range(len(event_ids) - 1)]
    return {"events": event_ids, "labels": labels, "order": order}


def _prefixes_of_spec(specification: dict[str, object]) -> set[PomsetKey]:
    if "sequence" in specification:
        specification = sequence_to_spec(specification["sequence"])
    event_ids = list(specification["events"])
    labels = dict(specification["labels"])
    immediate_preds = {event_id: set() for event_id in event_ids}
    for src, dst in specification.get("order", []):
        immediate_preds[dst].add(src)
    strict_order = _transitive_closure(event_ids, immediate_preds)
    family: set[PomsetKey] = set()
    for subset in _downward_closed_subsets(event_ids, strict_order):
        sub_labels = {event_id: labels[event_id] for event_id in subset}
        sub_order = {event_id: strict_order[event_id].intersection(subset) for event_id in subset}
        family.add(canonical_pomset_key(list(subset), sub_labels, sub_order))
    return family


def _downward_closed_subsets(
    event_ids: Sequence[str],
    strict_order: Mapping[str, set[str]],
) -> list[set[str]]:
    subsets: list[set[str]] = []
    for mask in range(1 << len(event_ids)):
        subset = {event_ids[index] for index in range(len(event_ids)) if (mask >> index) & 1}
        if all(strict_order[event_id].issubset(subset) for event_id in subset):
            subsets.append(subset)
    return subsets


def _transitive_closure(
    event_ids: Sequence[str],
    immediate_preds: Mapping[str, set[str]],
) -> dict[str, set[str]]:
    closure: dict[str, set[str]] = {event_id: set() for event_id in event_ids}

    def compute(event_id: str) -> set[str]:
        if closure[event_id]:
            return closure[event_id]
        ancestors = set(immediate_preds[event_id])
        for predecessor in immediate_preds[event_id]:
            ancestors.update(compute(predecessor))
        closure[event_id] = ancestors
        return ancestors

    for event_id in event_ids:
        compute(event_id)
    return closure


def _cover_relations(
    event_ids: Sequence[str],
    strict_order: Mapping[str, set[str]],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    cover_preds = {event_id: set() for event_id in event_ids}
    cover_succs = {event_id: set() for event_id in event_ids}
    for lower in event_ids:
        for upper in event_ids:
            if lower == upper or lower not in strict_order[upper]:
                continue
            is_cover = True
            for middle in event_ids:
                if middle in {lower, upper}:
                    continue
                if lower in strict_order[middle] and middle in strict_order[upper]:
                    is_cover = False
                    break
            if is_cover:
                cover_preds[upper].add(lower)
                cover_succs[lower].add(upper)
    return cover_preds, cover_succs


def _stable_colors(
    event_ids: Sequence[str],
    labels: Mapping[str, str],
    strict_order: Mapping[str, set[str]],
    descendants: Mapping[str, set[str]],
    cover_preds: Mapping[str, set[str]],
    cover_succs: Mapping[str, set[str]],
    individualized: Mapping[str, int] | None = None,
) -> dict[str, int]:
    individualized = individualized or {}
    signatures = {
        event_id: (
            individualized.get(event_id),
            labels[event_id],
            len(strict_order[event_id]),
            len(descendants[event_id]),
            tuple(sorted(Counter(labels[x] for x in strict_order[event_id]).items())),
            tuple(sorted(Counter(labels[x] for x in descendants[event_id]).items())),
        )
        for event_id in event_ids
    }
    colors = _compress(signatures)
    while True:
        refined = {
            event_id: (
                individualized.get(event_id),
                labels[event_id],
                tuple(sorted(Counter(colors[x] for x in strict_order[event_id]).items())),
                tuple(sorted(Counter(colors[x] for x in descendants[event_id]).items())),
                tuple(sorted(Counter(colors[x] for x in cover_preds[event_id]).items())),
                tuple(sorted(Counter(colors[x] for x in cover_succs[event_id]).items())),
            )
            for event_id in event_ids
        }
        new_colors = _compress(refined)
        if new_colors == colors:
            return colors
        colors = new_colors


def _compress(signatures: Mapping[str, object]) -> dict[str, int]:
    palette = {signature: index for index, signature in enumerate(sorted(set(signatures.values())))}
    return {event_id: palette[signature] for event_id, signature in signatures.items()}


def _class_signature(
    class_members: Sequence[str],
    labels: Mapping[str, str],
    strict_order: Mapping[str, set[str]],
    descendants: Mapping[str, set[str]],
) -> tuple[object, ...]:
    sample = class_members[0]
    return (
        labels[sample],
        len(strict_order[sample]),
        len(descendants[sample]),
        len(class_members),
    )


def _encode(
    event_order: Sequence[str],
    labels: Mapping[str, str],
    strict_order: Mapping[str, set[str]],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    label_tuple = tuple(labels[event_id] for event_id in event_order)
    bits: list[int] = []
    for left, right in combinations(range(len(event_order)), 2):
        left_id = event_order[left]
        right_id = event_order[right]
        bits.append(1 if left_id in strict_order[right_id] else 0)
        bits.append(1 if right_id in strict_order[left_id] else 0)
    return label_tuple, tuple(bits)


def _canonical_search(
    event_ids: tuple[str, ...],
    labels: Mapping[str, str],
    strict_order: Mapping[str, set[str]],
    descendants: Mapping[str, set[str]],
    cover_preds: Mapping[str, set[str]],
    cover_succs: Mapping[str, set[str]],
    individualized_items: tuple[tuple[str, int], ...],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    individualized = dict(individualized_items)
    colors = _stable_colors(
        event_ids,
        labels,
        strict_order,
        descendants,
        cover_preds,
        cover_succs,
        individualized=individualized,
    )
    classes: dict[int, list[str]] = defaultdict(list)
    for event_id in event_ids:
        classes[colors[event_id]].append(event_id)
    if all(len(class_members) == 1 for class_members in classes.values()):
        ordered = tuple(
            class_members[0]
            for _, class_members in sorted(
                classes.items(),
                key=lambda item: _class_signature(item[1], labels, strict_order, descendants),
            )
        )
        return _encode(ordered, labels, strict_order)
    ambiguous = min(
        (tuple(sorted(class_members)) for class_members in classes.values() if len(class_members) > 1),
        key=lambda class_members: (
            len(class_members),
            _class_signature(class_members, labels, strict_order, descendants),
            class_members,
        ),
    )
    next_tag_base = 0 if not individualized else max(individualized.values()) + 1
    best: tuple[tuple[str, ...], tuple[int, ...]] | None = None
    for offset, event_id in enumerate(ambiguous):
        refined = tuple(sorted(list(individualized.items()) + [(event_id, next_tag_base + offset)]))
        candidate = _canonical_search(
            event_ids,
            labels,
            strict_order,
            descendants,
            cover_preds,
            cover_succs,
            refined,
        )
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best


def _one_topological_order(size: int, order_pairs: Sequence[tuple[int, int]]) -> list[int]:
    preds = {index: set() for index in range(size)}
    succs = {index: set() for index in range(size)}
    for src, dst in order_pairs:
        preds[dst].add(src)
        succs[src].add(dst)
    available = sorted(index for index in range(size) if not preds[index])
    out: list[int] = []
    while available:
        chosen = available.pop(0)
        out.append(chosen)
        for successor in sorted(succs[chosen]):
            preds[successor].remove(chosen)
            if not preds[successor]:
                available.append(successor)
        available.sort()
    return out
