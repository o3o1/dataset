from __future__ import annotations

import argparse
import json
import random
import re
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class ComponentPin:
    component: str
    pin: str


@dataclass
class PlaceholderDefinition:
    type: str
    pins: Optional[List[str]] = None


@dataclass
class PinSpec:
    placeholder: str
    pins: List[str]
    role: Optional[str] = None


@dataclass
class NodeConstraint:
    node_role: str
    pins: List[PinSpec]


@dataclass
class PinMapping:
    module_pin: str
    placeholder: Optional[str] = None
    pin: Optional[str] = None
    pin_options: Optional[List[str]] = None
    node_role: Optional[str] = None
    reuse_role: Optional[str] = None
    exclude_roles: Optional[List[str]] = None


@dataclass
class ModuleDefinition:
    name: str
    placeholders: Dict[str, PlaceholderDefinition]
    constraints: List[NodeConstraint]
    pin_mappings: List[PinMapping]
    module_type: str = "atomic"


@dataclass
class MatchedModule:
    definition: ModuleDefinition
    placeholder_assignments: Dict[str, str]
    node_nets: Dict[str, str]
    instance_name: str
    role_pins: Dict[str, Dict[str, str]]
    constraint_pin_usage: Dict[str, Set[str]]


@dataclass
class AtomicSampleSpec:
    module_count: int
    preferred_module: Optional[str] = None


def load_module_definitions(path: Path) -> List[ModuleDefinition]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    modules = []
    for module in raw.get("modules", []):
        placeholders: Dict[str, PlaceholderDefinition] = {}
        for item in module.get("placeholders", []):
            pins_field = item.get("pins")
            pins = [str(pin) for pin in pins_field] if pins_field is not None else None
            placeholders[item["name"]] = PlaceholderDefinition(
                type=item["type"],
                pins=pins,
            )

        constraints = []
        for constraint in module.get("constraints", []):
            pin_specs: List[PinSpec] = []
            for pin_entry in constraint.get("pins", []):
                pin_options_field = pin_entry.get("pins") or pin_entry.get("pin_options")
                if pin_options_field is None:
                    if "pin" not in pin_entry:
                        raise ValueError(
                            f"Constraint entry missing 'pin' or 'pins': {pin_entry}"
                        )
                    pin_options = [str(pin_entry["pin"])]
                else:
                    pin_options = [str(option) for option in pin_options_field]

                role = pin_entry.get("role") or pin_entry.get("pin_role")
                if role is None and len(pin_options) == 1:
                    role = str(pin_options[0])

                pin_specs.append(
                    PinSpec(
                        placeholder=pin_entry["placeholder"],
                        pins=pin_options,
                        role=role,
                    )
                )

            constraints.append(NodeConstraint(node_role=constraint["node_role"], pins=pin_specs))

        pin_mappings = []
        for mapping in module.get("pin_mappings", []):
            pin_options_field = mapping.get("pin_options") or mapping.get("pins")
            pin_options = (
                [str(option) for option in pin_options_field]
                if pin_options_field is not None
                else None
            )

            exclude_roles_field = mapping.get("exclude_roles")
            exclude_roles = (
                [str(role) for role in exclude_roles_field]
                if exclude_roles_field is not None
                else None
            )

            pin_mappings.append(
                PinMapping(
                    module_pin=str(mapping["module_pin"]),
                    placeholder=mapping.get("placeholder"),
                    pin=str(mapping.get("pin")) if mapping.get("pin") is not None else None,
                    pin_options=pin_options,
                    node_role=mapping.get("node_role"),
                    reuse_role=mapping.get("reuse_role"),
                    exclude_roles=exclude_roles,
                )
            )

        modules.append(
            ModuleDefinition(
                name=module["name"],
                placeholders=placeholders,
                constraints=constraints,
                pin_mappings=pin_mappings,
            )
        )

    classify_module_types(modules)

    return modules


def classify_module_types(modules: List[ModuleDefinition]) -> None:
    module_names = {module.name for module in modules}

    for module in modules:
        total_placeholders = len(module.placeholders)
        module_placeholder_count = sum(
            1
            for placeholder in module.placeholders.values()
            if placeholder.type in module_names
        )

        if total_placeholders == 0 or module_placeholder_count == 0:
            module.module_type = "atomic"
        elif module_placeholder_count == total_placeholders:
            module.module_type = "composite"
        else:
            module.module_type = "hybrid"


def infer_component_type(component_id: str) -> str:
    if "@" in component_id:
        return component_id.split("@", 1)[0]
    match = re.match(r"([A-Za-z]+)", component_id)
    return match.group(1) if match else component_id


def parse_netlist(lines: Iterable[str]) -> Tuple[Dict[str, List[ComponentPin]], Dict[str, Dict[str, str]]]:
    nets: Dict[str, List[ComponentPin]] = {}
    component_pins: Dict[str, Dict[str, str]] = defaultdict(dict)

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        net_name = parts[0]
        pins: List[ComponentPin] = []
        for token in parts[1:]:
            if "(" not in token or not token.endswith(")"):
                raise ValueError(f"Invalid pin token '{token}' in line '{line}'")
            component = token[: token.index("(")]
            pin = token[token.index("(") + 1 : -1]
            pins.append(ComponentPin(component=component, pin=pin))
            component_pins[component][pin] = net_name

        nets[net_name] = pins

    return nets, component_pins


def build_component_index(component_pins: Dict[str, Dict[str, str]]) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = defaultdict(list)
    for component in component_pins:
        index[infer_component_type(component)].append(component)
    return index


def collect_placeholder_pins(definition: ModuleDefinition) -> Dict[str, List[str]]:
    placeholder_pins: Dict[str, Set[str]] = defaultdict(set)
    for name, placeholder in definition.placeholders.items():
        if placeholder.pins:
            placeholder_pins[name].update(placeholder.pins)

    for constraint in definition.constraints:
        for pin in constraint.pins:
            placeholder_pins[pin.placeholder].update(pin.pins)

    for mapping in definition.pin_mappings:
        if mapping.placeholder:
            if mapping.pin is not None:
                placeholder_pins[mapping.placeholder].add(mapping.pin)
            if mapping.pin_options:
                placeholder_pins[mapping.placeholder].update(mapping.pin_options)

    return {placeholder: sorted(pins) for placeholder, pins in placeholder_pins.items()}


def satisfy_constraints(
    definition: ModuleDefinition,
    assignments: Dict[str, str],
    component_pins: Dict[str, Dict[str, str]],
) -> Optional[Tuple[Dict[str, str], Dict[str, Dict[str, str]], Dict[str, Set[str]]]]:
    node_nets: Dict[str, str] = {}
    role_selection: Dict[Tuple[str, Optional[str]], str] = {}
    used_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    def assign_constraint(index: int) -> bool:
        if index == len(definition.constraints):
            return True

        constraint = definition.constraints[index]
        target_net = node_nets.get(constraint.node_role)

        def assign_pin_spec(pin_idx: int, current_net: Optional[str]) -> bool:
            if pin_idx == len(constraint.pins):
                if current_net is None:
                    return False
                if target_net is not None and current_net != target_net:
                    return False

                previous = node_nets.get(constraint.node_role)
                node_nets[constraint.node_role] = current_net
                if assign_constraint(index + 1):
                    return True
                if previous is None:
                    node_nets.pop(constraint.node_role, None)
                else:
                    node_nets[constraint.node_role] = previous
                return False

            pin_spec = constraint.pins[pin_idx]
            placeholder = pin_spec.placeholder
            component_id = assignments.get(placeholder)
            if component_id is None:
                return False

            component_netmap = component_pins.get(component_id, {})
            role_key = (placeholder, pin_spec.role) if pin_spec.role is not None else None

            if role_key and role_key in role_selection:
                candidate_pins = [role_selection[role_key]]
            else:
                candidate_pins = [pin for pin in pin_spec.pins if pin in component_netmap]

            if not candidate_pins:
                return False

            for pin in candidate_pins:
                if used_counts.get(placeholder, Counter()).get(pin):
                    continue
                net = component_netmap.get(pin)
                if net is None:
                    continue
                if current_net is not None and net != current_net:
                    continue

                added_role = False
                if role_key and role_key not in role_selection:
                    role_selection[role_key] = pin
                    added_role = True

                used_counts[placeholder][pin] += 1

                next_net = net if current_net is None else current_net
                if assign_pin_spec(pin_idx + 1, next_net):
                    return True

                used_counts[placeholder][pin] -= 1
                if used_counts[placeholder][pin] == 0:
                    del used_counts[placeholder][pin]
                if not used_counts[placeholder]:
                    used_counts.pop(placeholder)

                if added_role:
                    role_selection.pop(role_key, None)

            return False

        return assign_pin_spec(0, target_net)

    if assign_constraint(0):
        role_map: Dict[str, Dict[str, str]] = defaultdict(dict)
        for (placeholder, role), pin in role_selection.items():
            if role is not None:
                role_map[placeholder][role] = pin

        used_map: Dict[str, Set[str]] = {
            placeholder: set(counter.keys()) for placeholder, counter in used_counts.items()
        }

        return node_nets.copy(), {k: dict(v) for k, v in role_map.items()}, used_map

    return None


def enumerate_assignments(
    definition: ModuleDefinition,
    component_index: Dict[str, List[str]],
    component_pins: Dict[str, Dict[str, str]],
    used_components: set,
) -> List[Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, str]], Dict[str, Set[str]]]]:
    placeholders = list(definition.placeholders.keys())
    results: List[Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, str]], Dict[str, Set[str]]]] = []

    def backtrack(idx: int, current: Dict[str, str]):
        if idx == len(placeholders):
            constraint_result = satisfy_constraints(definition, current, component_pins)
            if constraint_result is not None:
                node_nets, role_pins, used_pins = constraint_result
                results.append((current.copy(), node_nets, role_pins, used_pins))
            return

        placeholder = placeholders[idx]
        required_type = definition.placeholders[placeholder].type
        for component in component_index.get(required_type, []):
            if component in used_components or component in current.values():
                continue
            if required_type != infer_component_type(component):
                continue
            current[placeholder] = component
            backtrack(idx + 1, current)
            current.pop(placeholder, None)

    backtrack(0, {})
    return results


def remove_components_from_nets(nets: Dict[str, List[ComponentPin]], components: Iterable[str]) -> None:
    components_set = set(components)
    for net_name, pins in nets.items():
        nets[net_name] = [pin for pin in pins if pin.component not in components_set]


def add_module_to_nets(
    nets: Dict[str, List[ComponentPin]],
    module: MatchedModule,
    component_pins: Dict[str, Dict[str, str]],
) -> Dict[str, str]:
    pin_nets: Dict[str, str] = {}
    placeholder_mapping_used: Dict[str, Set[str]] = defaultdict(set)

    for mapping in module.definition.pin_mappings:
        if mapping.node_role:
            net = module.node_nets[mapping.node_role]
        elif mapping.placeholder:
            placeholder = mapping.placeholder
            component_id = module.placeholder_assignments[placeholder]
            component_netmap = component_pins[component_id]
            role_map = module.role_pins.get(placeholder, {})

            pin: Optional[str] = None
            if mapping.reuse_role:
                pin = role_map.get(mapping.reuse_role)
                if pin is None:
                    raise ValueError(
                        f"Role '{mapping.reuse_role}' not assigned for placeholder '{placeholder}'"
                    )
            elif mapping.pin is not None:
                pin = mapping.pin
            else:
                candidates = set(component_netmap.keys())
                if mapping.pin_options:
                    candidates &= set(mapping.pin_options)
                if mapping.exclude_roles:
                    excluded = {
                        role_map[role]
                        for role in mapping.exclude_roles
                        if role in role_map
                    }
                    candidates -= excluded
                candidates -= placeholder_mapping_used.get(placeholder, set())
                if not candidates:
                    raise ValueError(
                        f"No available pins for module pin {mapping.module_pin} on placeholder '{placeholder}'"
                    )
                pin = sorted(candidates)[0]

            if pin not in component_netmap:
                raise ValueError(
                    f"Component '{component_id}' lacks pin '{pin}' needed by module mapping"
                )

            if not mapping.reuse_role:
                placeholder_mapping_used.setdefault(placeholder, set()).add(pin)

            net = component_netmap[pin]
        else:
            raise ValueError("Pin mapping requires either node_role or placeholder")

        pin_nets[mapping.module_pin] = net
        nets[net].append(ComponentPin(component=module.instance_name, pin=mapping.module_pin))

    return pin_nets


def reconstruct_netlist(nets: Dict[str, List[ComponentPin]]) -> List[str]:
    lines: List[str] = []
    for net_name, pin_refs in nets.items():
        pins = " ".join(f"{pin.component}({pin.pin})" for pin in pin_refs)
        line = f"{net_name} {pins}" if pins else net_name
        lines.append(line.strip())
    return lines


def process_netlist(
    lines: Iterable[str],
    module_definitions: List[ModuleDefinition],
) -> Dict[str, object]:
    nets, component_pins = parse_netlist(lines)
    component_index = build_component_index(component_pins)
    working_nets = {net: pins.copy() for net, pins in nets.items()}

    used_components: set = set()
    type_counters: Dict[str, int] = defaultdict(int)
    modules_payload: List[Dict[str, object]] = []

    for definition in module_definitions:
        assignments = enumerate_assignments(definition, component_index, component_pins, used_components)
        for placeholder_assignments, node_nets, role_pins, used_pins in assignments:
            if any(component in used_components for component in placeholder_assignments.values()):
                continue

            type_counters[definition.name] += 1
            instance_name = f"{definition.name}@{type_counters[definition.name]}"

            module = MatchedModule(
                definition=definition,
                placeholder_assignments=placeholder_assignments,
                node_nets=node_nets,
                instance_name=instance_name,
                role_pins=role_pins,
                constraint_pin_usage=used_pins,
            )

            used_components.update(placeholder_assignments.values())
            remove_components_from_nets(working_nets, placeholder_assignments.values())

            pin_nets = add_module_to_nets(working_nets, module, component_pins)

            component_pins[module.instance_name] = pin_nets.copy()
            component_index.setdefault(definition.name, []).append(module.instance_name)

            modules_payload.append(
                {
                    "module_name": module.definition.name,
                    "instance_name": module.instance_name,
                    "placeholders": module.placeholder_assignments,
                    "node_nets": module.node_nets,
                    "pin_nets": pin_nets,
                    "module_type": module.definition.module_type,
                    "role_pins": {k: v for k, v in module.role_pins.items()},
                    "constraint_pins": {
                        placeholder: sorted(pins)
                        for placeholder, pins in module.constraint_pin_usage.items()
                    },
                }
            )

    replaced_netlist = reconstruct_netlist(working_nets)

    return {
        "modules": modules_payload,
        "replaced_netlist": replaced_netlist,
    }


def normalize_netlist_field(netlist_field: object) -> List[str]:
    if isinstance(netlist_field, list):
        return [str(line) for line in netlist_field]
    if isinstance(netlist_field, str):
        return [line for line in netlist_field.splitlines() if line.strip()]
    raise TypeError("netlist field must be a string or list of strings")


def parse_type_ratio(value: str) -> Dict[str, float]:
    default_ratio = {
        "atomic": 1.0,
        "hybrid": 1.0,
        "composite": 1.0,
    }

    if not value:
        return default_ratio

    ratio = default_ratio.copy()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid type ratio segment '{part}', expected format type=value")
        key, val = part.split("=", 1)
        key = key.strip()
        try:
            ratio[key] = float(val)
        except ValueError as exc:
            raise ValueError(f"Invalid ratio value for '{key}': {val}") from exc
    return ratio


def parse_category_ratio(value: str) -> Dict[str, float]:
    default_ratio = {
        "atomic": 1.0,
        "atomic_hybrid": 1.0,
        "atomic_hybrid_composite": 1.0,
    }

    if not value:
        return default_ratio

    ratio = default_ratio.copy()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid category ratio segment '{part}', expected format category=value")
        key, val = part.split("=", 1)
        key = key.strip()
        try:
            ratio[key] = float(val)
        except ValueError as exc:
            raise ValueError(f"Invalid ratio value for '{key}': {val}") from exc
    return ratio


def parse_ratio_map(value: str, default_ratio: Dict[str, float]) -> Dict[str, float]:
    if not value:
        return default_ratio.copy()

    ratio = default_ratio.copy()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid ratio segment '{part}', expected format key=value")
        key, val = part.split("=", 1)
        key = key.strip()
        try:
            ratio[key] = float(val)
        except ValueError as exc:
            raise ValueError(f"Invalid ratio value for '{key}': {val}") from exc
    return ratio


def allocate_counts_from_ratio(
    total: int,
    ratio: Dict[str, float],
    keys: Sequence[str],
) -> Dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")

    if total == 0:
        return {key: 0 for key in keys}

    weights = [max(ratio.get(key, 0.0), 0.0) for key in keys]
    weight_sum = sum(weights)
    if weight_sum == 0:
        weights = [1.0 for _ in keys]
        weight_sum = float(len(keys))

    raw_counts = [total * weight / weight_sum for weight in weights]
    base_counts = [int(math.floor(value)) for value in raw_counts]
    remainder = total - sum(base_counts)

    fractions = [value - base for value, base in zip(raw_counts, base_counts)]
    order = sorted(
        range(len(keys)),
        key=lambda idx: (fractions[idx], weights[idx]),
        reverse=True,
    )

    for idx in order[:remainder]:
        base_counts[idx] += 1

    return {key: base_counts[idx] for idx, key in enumerate(keys)}


def is_sample_valid(selected_types: Iterable[str], module_types: Iterable[str]) -> bool:
    selected_set = set(selected_types)
    actual_set = set(module_types)

    if not selected_set:
        return False

    if "composite" in selected_set:
        allowed_types = {"atomic", "hybrid", "composite"}
    elif "hybrid" in selected_set:
        allowed_types = {"atomic", "hybrid"}
    else:
        allowed_types = {"atomic"}

    return selected_set.issubset(actual_set) and actual_set.issubset(allowed_types)


def _copy_nets(
    nets: Dict[str, List[ComponentPin]]
) -> Dict[str, List[ComponentPin]]:
    return {
        net: [ComponentPin(component=pin.component, pin=pin.pin) for pin in pins]
        for net, pins in nets.items()
    }


def _component_pin_map(
    nets: Dict[str, List[ComponentPin]]
) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = defaultdict(dict)
    for net, pins in nets.items():
        for pin in pins:
            mapping[pin.component][pin.pin] = net
    return mapping


def _find_component_pin(
    nets: Dict[str, List[ComponentPin]],
    component: str,
    pin: str,
) -> Optional[Tuple[str, int]]:
    for net, pins in nets.items():
        for idx, pin_ref in enumerate(pins):
            if pin_ref.component == component and pin_ref.pin == pin:
                return net, idx
    return None


def _move_component_pin(
    nets: Dict[str, List[ComponentPin]],
    component: str,
    pin: str,
    new_net: str,
) -> bool:
    location = _find_component_pin(nets, component, pin)
    if not location:
        return False
    net_name, idx = location
    nets[new_net].append(ComponentPin(component=component, pin=pin))
    del nets[net_name][idx]
    if not nets[net_name]:
        del nets[net_name]
    return True


def _next_net_name(nets: Dict[str, List[ComponentPin]]) -> str:
    max_index = 0
    for net in nets.keys():
        match = re.fullmatch(r"/N(\d+)", net)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return f"/N{max_index + 1}"


def _has_unique_component_pins(nets: Dict[str, List[ComponentPin]]) -> bool:
    seen: Set[Tuple[str, str]] = set()
    for pins in nets.values():
        for pin in pins:
            key = (pin.component, pin.pin)
            if key in seen:
                return False
            seen.add(key)
    return True


def _isolate_all_components(
    nets: Dict[str, List[ComponentPin]]
) -> Dict[str, List[ComponentPin]]:
    component_map = _component_pin_map(nets)
    new_nets: Dict[str, List[ComponentPin]] = {}
    index = 1
    for component, pins in component_map.items():
        for pin in pins:
            net_name = f"/N{index}"
            index += 1
            new_nets[net_name] = [ComponentPin(component=component, pin=pin)]
    return new_nets


def _collect_module_critical_pins(module: Dict[str, object]) -> List[Tuple[str, str]]:
    critical: List[Tuple[str, str]] = []
    placeholders = module.get("placeholders", {})
    for placeholder, pins in module.get("constraint_pins", {}).items():
        component = placeholders.get(placeholder)
        if component is None:
            continue
        for pin in pins:
            critical.append((component, pin))
    if not critical:
        for component in placeholders.values():
            critical.extend((component, str(idx + 1)) for idx in range(2))
    return critical


def _break_single_module(
    nets: Dict[str, List[ComponentPin]],
    module: Dict[str, object],
    rng: random.Random,
) -> bool:
    return _detach_module_pins(nets, [module], rng)


def _detach_module_pins(
    nets: Dict[str, List[ComponentPin]],
    modules: List[Dict[str, object]],
    rng: random.Random,
) -> bool:
    if not modules:
        return False

    changed = False
    for module in modules:
        critical = _collect_module_critical_pins(module)
        rng.shuffle(critical)
        for component, pin in critical:
            if not _find_component_pin(nets, component, pin):
                continue
            new_net = _next_net_name(nets)
            nets.setdefault(new_net, [])
            if _move_component_pin(nets, component, pin, new_net):
                changed = True
                break
    return changed


def generate_negative_netlist(
    base_lines: List[str],
    module_definitions: List[ModuleDefinition],
    rng: random.Random,
    mutate_prob: float,
    invalidation_prob: float,
    max_mutations: int,
) -> List[str]:
    base_nets, _ = parse_netlist(base_lines)
    base_result = process_netlist(base_lines, module_definitions)
    modules = base_result.get("modules", [])

    attempts = max(max_mutations, 5)
    for _ in range(attempts):
        nets = _copy_nets(base_nets)

        success = False
        if modules:
            if len(modules) == 1 and modules[0].get("module_type") == "atomic":
                success = _break_single_module(nets, modules[0], rng)
            if not success:
                success = _detach_module_pins(nets, modules, rng)
        if not success:
            nets = _isolate_all_components(base_nets)
        if not _has_unique_component_pins(nets):
            continue

        mutated_lines = reconstruct_netlist(nets)
        mutated_result = process_netlist(mutated_lines, module_definitions)
        if not mutated_result.get("modules"):
            return mutated_lines

    isolated_nets = _isolate_all_components(base_nets)
    if not _has_unique_component_pins(isolated_nets):
        return []
    isolated_lines = reconstruct_netlist(isolated_nets)
    if not process_netlist(isolated_lines, module_definitions).get("modules"):
        return isolated_lines
    return []


def generate_random_netlist(
    module_definitions: List[ModuleDefinition],
    rng: random.Random,
    min_modules: int,
    max_modules: int,
    reuse_probability: float,
    type_ratio: Dict[str, float],
) -> List[str]:
    if not module_definitions:
        raise ValueError("No module definitions provided")
    if min_modules < 1:
        raise ValueError("min_modules must be at least 1")
    if max_modules < min_modules:
        raise ValueError("max_modules must be greater than or equal to min_modules")

    nets: Dict[str, List[ComponentPin]] = {}
    component_counters: Dict[str, int] = defaultdict(int)
    net_index = 1
    module_def_map = {definition.name: definition for definition in module_definitions}
    selected_types: List[str] = []

    def new_net_name() -> str:
        nonlocal net_index
        name = f"/N{net_index}"
        net_index += 1
        nets[name] = []
        return name

    def choose_net(
        excluded: Optional[Set[str]] = None,
        allow_reuse: bool = True,
    ) -> str:
        excluded_set = set(excluded) if excluded is not None else set()
        reusable = [name for name in nets.keys() if name not in excluded_set]
        if allow_reuse and reusable and rng.random() < reuse_probability:
            return rng.choice(reusable)
        return new_net_name()

    def new_component_id(component_type: str) -> str:
        component_counters[component_type] += 1
        return f"{component_type}{component_counters[component_type]}"

    def instantiate_module(
        definition: ModuleDefinition,
        preset_module_pin_nets: Optional[Dict[str, str]] = None,
        allow_cross_connections: bool = True,
    ) -> Dict[str, str]:
        placeholder_pin_map = collect_placeholder_pins(definition)
        placeholder_pin_to_net: Dict[str, Dict[str, str]] = {
            placeholder: {} for placeholder in placeholder_pin_map
        }
        placeholder_assigned_nets: Dict[str, Set[str]] = defaultdict(set)
        node_nets: Dict[str, str] = {}
        module_constraint_nets: Set[str] = set()
        module_free_nets: Set[str] = set()
        role_selection: Dict[Tuple[str, Optional[str]], str] = {}

        placeholder_pin_presets: Dict[str, Dict[str, str]] = defaultdict(dict)
        if preset_module_pin_nets:
            for mapping in definition.pin_mappings:
                net = preset_module_pin_nets.get(str(mapping.module_pin))
                if net is None:
                    continue
                if mapping.node_role:
                    node_nets[mapping.node_role] = net
                elif mapping.placeholder and mapping.pin is not None:
                    placeholder_pin_presets[mapping.placeholder][mapping.pin] = net

        for constraint in definition.constraints:
            net_name = node_nets.get(constraint.node_role)
            if net_name is None:
                net_name = choose_net(allow_reuse=allow_cross_connections)
                node_nets[constraint.node_role] = net_name
            module_constraint_nets.add(net_name)

            for pin_spec in constraint.pins:
                placeholder = pin_spec.placeholder
                assigned_pins = placeholder_pin_to_net.setdefault(placeholder, {})
                preset_map = placeholder_pin_presets.get(placeholder, {})
                candidate_pins = list(pin_spec.pins)
                rng.shuffle(candidate_pins)

                chosen_pin: Optional[str] = None
                role_key = (placeholder, pin_spec.role) if pin_spec.role is not None else None

                if role_key and role_key in role_selection:
                    pin = role_selection[role_key]
                    preset_net = preset_map.get(pin)
                    existing_net = assigned_pins.get(pin)
                    if ((preset_net is None or preset_net == net_name)
                        and (existing_net is None or existing_net == net_name)):
                        chosen_pin = pin

                if chosen_pin is None:
                    for pin in candidate_pins:
                        preset_net = preset_map.get(pin)
                        if preset_net is not None:
                            if preset_net == net_name:
                                chosen_pin = pin
                                break
                            continue

                if chosen_pin is None:
                    for pin in candidate_pins:
                        existing_net = assigned_pins.get(pin)
                        if existing_net is not None:
                            if existing_net == net_name:
                                chosen_pin = pin
                                break
                            continue
                        preset_net = preset_map.get(pin)
                        if preset_net is not None and preset_net != net_name:
                            continue
                        chosen_pin = pin
                        break

                if chosen_pin is None:
                    chosen_pin = candidate_pins[0]

                assigned_pins[chosen_pin] = net_name
                placeholder_assigned_nets[placeholder].add(net_name)
                if role_key and role_key not in role_selection:
                    role_selection[role_key] = chosen_pin

        for placeholder, pin_map in placeholder_pin_presets.items():
            assigned_pins = placeholder_pin_to_net.setdefault(placeholder, {})
            for pin, net in pin_map.items():
                if pin in assigned_pins:
                    continue
                assigned_pins[pin] = net
                placeholder_assigned_nets[placeholder].add(net)
                module_constraint_nets.add(net)

        for placeholder, pins in placeholder_pin_map.items():
            assigned_pins = placeholder_pin_to_net.setdefault(placeholder, {})
            for pin in pins:
                if pin in assigned_pins:
                    continue
                preset_net = placeholder_pin_presets.get(placeholder, {}).get(pin)
                if preset_net is not None:
                    net_name = preset_net
                else:
                    exclude_nets = (
                        placeholder_assigned_nets.get(placeholder, set())
                        .union(module_constraint_nets)
                        .union(module_free_nets)
                    )
                    net_name = choose_net(
                        excluded=exclude_nets,
                        allow_reuse=allow_cross_connections,
                    )
                    module_free_nets.add(net_name)
                assigned_pins[pin] = net_name
                placeholder_assigned_nets[placeholder].add(net_name)

        for placeholder, placeholder_def in definition.placeholders.items():
            pin_assignment = placeholder_pin_to_net.get(placeholder, {})
            component_type = placeholder_def.type
            if component_type in module_def_map:
                instantiate_module(
                    module_def_map[component_type],
                    preset_module_pin_nets=pin_assignment,
                    allow_cross_connections=allow_cross_connections,
                )
            else:
                component_id = new_component_id(component_type)
                for pin, net in pin_assignment.items():
                    nets.setdefault(net, []).append(
                        ComponentPin(component=component_id, pin=pin)
                    )

        module_pin_nets: Dict[str, str] = {}
        for mapping in definition.pin_mappings:
            if mapping.node_role:
                net = node_nets[mapping.node_role]
            elif mapping.placeholder:
                pin_assignments = placeholder_pin_to_net[mapping.placeholder]

                if mapping.reuse_role:
                    role_key = (mapping.placeholder, mapping.reuse_role)
                    pin_name = role_selection.get(role_key)
                    if pin_name is None:
                        raise ValueError(
                            f"Role '{mapping.reuse_role}' not assigned for placeholder '{mapping.placeholder}'"
                        )
                    net = pin_assignments[pin_name]
                elif mapping.pin is not None:
                    net = pin_assignments[mapping.pin]
                else:
                    candidates = set(pin_assignments.keys())
                    if mapping.pin_options:
                        candidates &= {str(option) for option in mapping.pin_options}
                    if mapping.exclude_roles:
                        excluded = {
                            role_selection[(mapping.placeholder, role)]
                            for role in mapping.exclude_roles
                            if (mapping.placeholder, role) in role_selection
                        }
                        candidates -= excluded
                    if not candidates:
                        raise ValueError(
                            f"No available pins for module pin {mapping.module_pin} on placeholder '{mapping.placeholder}'"
                        )
                    pin_name = sorted(candidates)[0]
                    net = pin_assignments[pin_name]
            else:
                raise ValueError("Pin mapping requires node_role or placeholder")
            module_pin_nets[str(mapping.module_pin)] = net

        return module_pin_nets

    definitions_by_type: Dict[str, List[ModuleDefinition]] = defaultdict(list)
    for definition in module_definitions:
        definitions_by_type[definition.module_type].append(definition)

    module_count = rng.randint(min_modules, max_modules)

    extra_atomic_max = 2

    for _ in range(module_count):
        available_types = [module_type for module_type, defs in definitions_by_type.items() if defs]
        if not available_types:
            raise ValueError("No available module definitions to instantiate")

        weights = []
        for module_type in available_types:
            weight = type_ratio.get(module_type, 0.0)
            weights.append(max(weight, 0.0))

        if all(weight == 0 for weight in weights):
            weights = [1.0 for _ in available_types]

        selected_type = rng.choices(available_types, weights=weights, k=1)[0]
        selected_types.append(selected_type)
        definition = rng.choice(definitions_by_type[selected_type])
        allow_cross = selected_type != "composite"
        instantiate_module(definition, allow_cross_connections=allow_cross)

        if selected_type in {"hybrid", "composite"} and definitions_by_type.get("atomic"):
            extra_count = rng.randint(0, extra_atomic_max)
            for _ in range(extra_count):
                extra_definition = rng.choice(definitions_by_type["atomic"])
                instantiate_module(extra_definition, allow_cross_connections=True)

    for net_name in list(nets.keys()):
        rng.shuffle(nets[net_name])

    return reconstruct_netlist(nets), selected_types


def build_atomic_task_list(
    total_samples: int,
    single_module_targets: Dict[str, int],
    module_count_targets: Dict[int, int],
    four_plus_choices: Sequence[int],
    seed: int,
) -> List[AtomicSampleSpec]:
    tasks: List[AtomicSampleSpec] = []

    for module_name, count in single_module_targets.items():
        tasks.extend(
            AtomicSampleSpec(module_count=1, preferred_module=module_name)
            for _ in range(count)
        )

    for module_count, count in module_count_targets.items():
        if module_count == 1:
            continue
        tasks.extend(AtomicSampleSpec(module_count=module_count) for _ in range(count))

    rng = random.Random(seed)
    assigned = sum(module_count_targets.values())
    four_plus_total = total_samples - assigned
    for _ in range(four_plus_total):
        tasks.append(AtomicSampleSpec(module_count=rng.choice(four_plus_choices)))

    rng.shuffle(tasks)
    return tasks


def ensure_atomic_modules(modules: Sequence[Dict[str, object]]) -> bool:
    return all(module.get("module_type") == "atomic" for module in modules)


def add_atomic_noise_components(netlist_lines: List[str], rng: random.Random) -> List[str]:
    nets, component_pins = parse_netlist(netlist_lines)

    existing_components = set(component_pins.keys())
    existing_nets = set(nets.keys())

    net_index = 0
    net_pattern = re.compile(r"/N(\d+)")
    for net_name in existing_nets:
        match = net_pattern.fullmatch(net_name)
        if match:
            net_index = max(net_index, int(match.group(1)))

    noise_prefixes = ("S", "D", "L", "C")
    prefix_next_index: Dict[str, int] = {prefix: 1 for prefix in noise_prefixes}
    component_pattern = re.compile(r"([A-Za-z]+)(\d+)$")
    for name in existing_components:
        match = component_pattern.match(name)
        if not match:
            continue
        prefix, number = match.groups()
        if prefix in prefix_next_index:
            prefix_next_index[prefix] = max(prefix_next_index[prefix], int(number) + 1)

    def new_component_name(prefix: str) -> str:
        index = prefix_next_index.get(prefix, 1)
        while True:
            candidate = f"{prefix}{index}"
            if candidate not in existing_components:
                existing_components.add(candidate)
                prefix_next_index[prefix] = index + 1
                return candidate
            index += 1

    def new_net_name() -> str:
        nonlocal net_index
        net_index += 1
        name = f"/N{net_index}"
        existing_nets.add(name)
        return name

    noise_count = rng.choice([1, 2])
    augmented_lines = list(netlist_lines)

    for _ in range(noise_count):
        prefix = rng.choice(noise_prefixes)
        component = new_component_name(prefix)
        for pin_idx in range(1, 3):
            net_name = new_net_name()
            augmented_lines.append(f"{net_name} {component}({pin_idx})")

    return augmented_lines


def run_convert(args: argparse.Namespace) -> None:
    module_definitions = load_module_definitions(args.definitions)

    with args.netlists.open("r", encoding="utf-8") as f_in, args.output.open("w", encoding="utf-8") as f_out:
        for raw_line in f_in:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = json.loads(raw_line)
            netlist_lines = normalize_netlist_field(record.get(args.netlist_field))
            processing_result = process_netlist(netlist_lines, module_definitions)

            output_entry = {
                "input": {"netlist": netlist_lines},
                "output": processing_result,
            }

            for field in args.copy_fields:
                if field in record:
                    output_entry["input"][field] = record[field]

            f_out.write(json.dumps(output_entry, ensure_ascii=False) + "\n")


def run_random(args: argparse.Namespace) -> None:
    module_definitions = load_module_definitions(args.definitions)
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    type_ratio = parse_type_ratio(args.type_ratio)

    entries: List[Dict[str, object]] = []
    with args.output.open("w", encoding="utf-8") as f_out:
        generated = 0
        while generated < args.samples:
            success = False
            last_error: Optional[str] = None
            for _ in range(max(args.max_retries, 1)):
                netlist_lines, selected_types = generate_random_netlist(
                    module_definitions=module_definitions,
                    rng=rng,
                    min_modules=args.min_modules,
                    max_modules=args.max_modules,
                    reuse_probability=args.reuse_prob,
                    type_ratio=type_ratio,
                )
                processing_result = process_netlist(netlist_lines, module_definitions)
                modules = processing_result["modules"]
                module_types = [module["module_type"] for module in modules]
                if is_sample_valid(selected_types, module_types):
                    entry = {
                        "input": {"netlist": netlist_lines},
                        "output": processing_result,
                    }
                    entry["output"]["selected_types"] = selected_types
                    entries.append(entry)
                    f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    generated += 1
                    success = True
                    break
                last_error = (
                    f"Generated sample contains module types {sorted(set(module_types))} "
                    f"which violates target types {sorted(set(selected_types))}"
                )
            if not success:
                raise RuntimeError(
                    last_error
                    or "Failed to generate a sample satisfying module type constraints"
                )

    if args.negatives:
        neg_dir = args.negatives
        neg_dir.mkdir(parents=True, exist_ok=True)

        neg_path = neg_dir / args.output.name
        with neg_path.open("w", encoding="utf-8") as f_neg:
            rng_neg = random.Random(args.seed)
            for entry in entries:
                base_netlist = entry["input"]["netlist"]
                negative_lines = generate_negative_netlist(
                    base_lines=base_netlist,
                    module_definitions=module_definitions,
                    rng=rng_neg,
                    mutate_prob=args.neg_mutate_prob,
                    invalidation_prob=args.neg_invalidation_prob,
                    max_mutations=args.neg_max_mutations,
                )
                if not negative_lines:
                    negative_lines = base_netlist
                negative_entry = {
                    "input": {"netlist": negative_lines},
                    "output": {
                        "modules": [],
                        "replaced_netlist": negative_lines,
                    },
                }
                f_neg.write(json.dumps(negative_entry, ensure_ascii=False) + "\n")


def run_atomic(args: argparse.Namespace) -> None:
    module_definitions = load_module_definitions(args.definitions)

    total_samples = args.samples
    if total_samples < 0:
        raise ValueError("--samples 必须为非负整数")

    default_module_count_ratio = {
        "1": 0.55,
        "2": 0.30,
        "3": 0.12,
        "4+": 0.03,
    }
    module_count_ratio = parse_ratio_map(args.module_count_ratio, default_module_count_ratio)
    module_count_distribution = allocate_counts_from_ratio(
        total_samples,
        module_count_ratio,
        ["1", "2", "3", "4+"],
    )
    module_count_targets = {
        1: module_count_distribution.get("1", 0),
        2: module_count_distribution.get("2", 0),
        3: module_count_distribution.get("3", 0),
    }

    default_single_module_ratio = {
        "rectifier_type1": 0.35,
        "rectifier_type2": 0.25,
        "halfbridge": 0.25,
        "filter": 0.15,
    }
    single_module_ratio = parse_ratio_map(args.single_module_ratio, default_single_module_ratio)
    single_module_targets = allocate_counts_from_ratio(
        module_count_targets.get(1, 0),
        single_module_ratio,
        list(single_module_ratio.keys()),
    )

    four_plus_tokens = [token.strip() for token in args.four_plus_options.split(",") if token.strip()]
    if not four_plus_tokens:
        raise ValueError("--four-plus-options 至少需要一个有效的整数")
    try:
        four_plus_choices = [int(token) for token in four_plus_tokens]
    except ValueError as exc:
        raise ValueError("--four-plus-options 必须全部为整数") from exc
    if any(choice < 4 for choice in four_plus_choices):
        raise ValueError("--four-plus-options 中的值必须不小于 4")

    tasks = build_atomic_task_list(
        total_samples=total_samples,
        single_module_targets=single_module_targets,
        module_count_targets=module_count_targets,
        four_plus_choices=four_plus_choices,
        seed=args.task_seed,
    )

    rng = random.Random(args.seed)
    type_ratio = {"atomic": 1.0, "hybrid": 0.0, "composite": 0.0}

    entries: List[Dict[str, object]] = []
    for index, task in enumerate(tasks, 1):
        for attempt in range(1, args.max_attempts + 1):
            netlist_lines, _ = generate_random_netlist(
                module_definitions=module_definitions,
                rng=rng,
                min_modules=task.module_count,
                max_modules=task.module_count,
                reuse_probability=args.reuse_prob,
                type_ratio=type_ratio,
            )

            processing_result = process_netlist(netlist_lines, module_definitions)
            modules = processing_result["modules"]

            if len(modules) != task.module_count:
                continue
            if not ensure_atomic_modules(modules):
                continue
            if task.preferred_module is not None:
                if len(modules) != 1 or modules[0]["module_name"] != task.preferred_module:
                    continue

            augmented_netlist = add_atomic_noise_components(netlist_lines, rng)
            augmented_result = process_netlist(augmented_netlist, module_definitions)
            augmented_modules = augmented_result["modules"]

            if len(augmented_modules) != task.module_count:
                continue
            if not ensure_atomic_modules(augmented_modules):
                continue

            entry = {
                "input": {"netlist": augmented_netlist},
                "output": augmented_result,
            }
            entries.append(entry)
            break
        else:
            raise RuntimeError(
                f"无法生成满足要求的样本（编号 {index}，目标模块数 {task.module_count}）"
            )

    if len(entries) != total_samples:
        raise RuntimeError(
            f"生成数量不匹配，期望 {total_samples} 实际 {len(entries)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Generate SFT dataset entries from netlists and module definitions")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="从JSONL网表转换数据集")
    convert_parser.add_argument("--definitions", type=Path, required=True, help="模块定义JSON文件路径")
    convert_parser.add_argument("--netlists", type=Path, required=True, help="输入网表JSONL路径")
    convert_parser.add_argument("--output", type=Path, required=True, help="输出JSONL路径")
    convert_parser.add_argument("--netlist-field", default="netlist", help="JSON记录中的网表字段名")
    convert_parser.add_argument("--copy-fields", nargs="*", default=[], help="需要复制到输入中的其他字段")
    convert_parser.set_defaults(handler=run_convert)

    random_parser = subparsers.add_parser("random", help="随机生成网表数据集")
    random_parser.add_argument("--definitions", type=Path, required=True, help="模块定义JSON文件路径")
    random_parser.add_argument("--output", type=Path, required=True, help="输出JSONL路径")
    random_parser.add_argument("--samples", type=int, default=100, help="随机样本数量")
    random_parser.add_argument("--min-modules", type=int, default=1, help="单个网表最少模块数")
    random_parser.add_argument("--max-modules", type=int, default=3, help="单个网表最多模块数")
    random_parser.add_argument("--reuse-prob", type=float, default=0.5, help="重用已有节点的概率")
    random_parser.add_argument("--seed", type=int, default=None, help="随机种子，便于复现")
    random_parser.add_argument(
        "--type-ratio",
        default="atomic=1,hybrid=1,composite=1",
        help="指定三类模块的采样权重，例如 atomic=2,hybrid=1,composite=1",
    )
    random_parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="为满足类型约束生成单条样本的最大重试次数",
    )
    random_parser.add_argument(
        "--negatives",
        type=Path,
        help="若指定，则在该目录下生成对应的负例数据集",
    )
    random_parser.add_argument(
        "--neg-mutate-prob",
        type=float,
        default=0.2,
        help="负例生成时执行各类变换的基础概率",
    )
    random_parser.add_argument(
        "--neg-invalidation-prob",
        type=float,
        default=0.3,
        help="额外强制插入明显错误结构的概率",
    )
    random_parser.add_argument(
        "--neg-max-mutations",
        type=int,
        default=5,
        help="单条负例的最大变换次数",
    )
    random_parser.set_defaults(handler=run_random)

    atomic_parser = subparsers.add_parser(
        "atomic",
        help="生成仅包含 atomic 模块且带少量噪声器件的数据集",
    )
    atomic_parser.add_argument(
        "--definitions", type=Path, required=True, help="模块定义JSON文件路径"
    )
    atomic_parser.add_argument(
        "--output", type=Path, required=True, help="输出JSONL路径"
    )
    atomic_parser.add_argument(
        "--samples", type=int, default=12_750, help="目标样本总数"
    )
    atomic_parser.add_argument(
        "--seed", type=int, default=2024, help="随机种子（影响模块生成与噪声）"
    )
    atomic_parser.add_argument(
        "--task-seed",
        type=int,
        default=4_294_967_291,
        help="任务排列的随机种子",
    )
    atomic_parser.add_argument(
        "--reuse-prob",
        type=float,
        default=0.5,
        help="生成网表时重用节点的概率",
    )
    atomic_parser.add_argument(
        "--module-count-ratio",
        default="1=0.55,2=0.30,3=0.12,4+=0.03",
        help="不同模块数量的目标比例，例如 1=0.55,2=0.30,3=0.12,4+=0.03",
    )
    atomic_parser.add_argument(
        "--single-module-ratio",
        default="rectifier_type1=0.35,rectifier_type2=0.25,halfbridge=0.25,filter=0.15",
        help="单模块样本中各模块类型的比例",
    )
    atomic_parser.add_argument(
        "--four-plus-options",
        default="4,5,6",
        help="当模块数≥4时允许的具体模块数量，逗号分隔",
    )
    atomic_parser.add_argument(
        "--max-attempts",
        type=int,
        default=5_000,
        help="为每个样本寻找满足约束的最大尝试次数",
    )
    atomic_parser.set_defaults(handler=run_atomic)

    args = parser.parse_args()

    if args.command == "random":
        if args.min_modules < 1:
            parser.error("--min-modules 必须不小于 1")
        if args.max_modules < args.min_modules:
            parser.error("--max-modules 必须大于等于 --min-modules")
        if not (0.0 <= args.reuse_prob <= 1.0):
            parser.error("--reuse-prob 必须在 [0, 1] 区间内")

    args.handler(args)


if __name__ == "__main__":
    run_cli()
