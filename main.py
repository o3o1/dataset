from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


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


def generate_negative_netlist(
    base_lines: List[str],
    module_definitions: List[ModuleDefinition],
    rng: random.Random,
    mutate_prob: float,
    invalidation_prob: float,
    max_mutations: int,
) -> List[str]:
    nets, component_pins = parse_netlist(base_lines)

    net_names = list(nets.keys())
    components = list(component_pins.keys())

    mutations = rng.randint(1, max_mutations)
    for _ in range(mutations):
        action = rng.random()
        if action < mutate_prob and net_names:
            net = rng.choice(net_names)
            if nets[net]:
                pin = rng.choice(nets[net])
                nets[net].remove(pin)
                if not nets[net]:
                    net_names.remove(net)
            continue

        if action < mutate_prob * 2 and components:
            component = rng.choice(components)
            pin_ids = list(component_pins.get(component, {}).keys())
            if not pin_ids:
                continue
            pin = rng.choice(pin_ids)
            net = component_pins[component][pin]
            nets.setdefault(net, [])
            nets[net] = [p for p in nets[net] if p.component != component or p.pin != pin]
            new_net = f"/NX{rng.randint(1, 999999)}"
            nets.setdefault(new_net, []).append(ComponentPin(component=component, pin=pin))
            component_pins[component][pin] = new_net
            if new_net not in net_names:
                net_names.append(new_net)
            continue

        if action < mutate_prob * 3 and components:
            component = rng.choice(components)
            pin_ids = list(component_pins[component].keys()) or ["1", "2", "3"]
            new_pin = rng.choice(pin_ids)
            source_net = component_pins[component].get(new_pin)
            target_net = rng.choice(net_names) if net_names else f"/NX{rng.randint(1, 999999)}"
            nets.setdefault(target_net, []).append(ComponentPin(component=component, pin=new_pin))
            component_pins[component][new_pin] = target_net
            if source_net and source_net in nets:
                nets[source_net] = [p for p in nets[source_net] if p.component != component or p.pin != new_pin]
            if target_net not in net_names:
                net_names.append(target_net)
            continue

        if action < mutate_prob * 4 and module_definitions:
            definition = rng.choice(module_definitions)
            placeholder_names = list(definition.placeholders.keys())
            if len(placeholder_names) < 2:
                continue
            ph_a, ph_b = rng.sample(placeholder_names, 2)
            def_type_a = definition.placeholders[ph_a].type
            def_type_b = definition.placeholders[ph_b].type
            nets.setdefault(f"/NX{rng.randint(1, 999999)}", [])
            nets.setdefault(f"/NX{rng.randint(1, 999999)}", [])
            for module in list(nets.values()):
                new_module = []
                for pin in module:
                    component = pin.component
                    if component.startswith(def_type_a):
                        component = component.replace(def_type_a, def_type_b, 1)
                    elif component.startswith(def_type_b):
                        component = component.replace(def_type_b, def_type_a, 1)
                    new_module.append(ComponentPin(component=component, pin=pin.pin))
                module[:] = new_module
            continue

        if action < mutate_prob * 5:
            net = f"/NX{rng.randint(1, 999999)}"
            nets[net] = []
            net_names.append(net)

    if rng.random() < invalidation_prob:
        net = f"/NX{rng.randint(1, 999999)}"
        nets[net] = [ComponentPin(component="BAD", pin="1")]

    lines = []
    for net_name, pins in nets.items():
        if not pins:
            continue
        pins_str = " ".join(f"{pin.component}({pin.pin})" for pin in pins)
        lines.append(f"{net_name} {pins_str}")
    return lines


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
                    "output": {"label": "negative"},
                }
                f_neg.write(json.dumps(negative_entry, ensure_ascii=False) + "\n")


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
