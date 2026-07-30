"""Canonical Atlas Generator delivery evidence for one target SPM."""

SCHEMA_VERSION = 1
DELIVERY_MODE_ASSET_REGISTRATION_ONLY = "asset_registration_only"
DELIVERY_MODE_RENDER_CONNECTED = "render_connected"
DELIVERY_MODE_CONNECTION_INCOMPLETE = "connection_incomplete"
GENERATOR_VARIANT_POLICY = "ensure_all_material_cutouts"


def positive_int(value):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _truthy(value):
    return str(value or "").strip().casefold() in {"1", "true", "yes"}


def _generator_type(value):
    return " ".join(
        "".join(
            character if character.isalnum() else " "
            for character in str(value or "").casefold()
        ).split()
    )


def _is_leaf_generator(value):
    normalized = _generator_type(value)
    return (
        normalized == "frond"
        or normalized.replace(" ", "") == "leafmesh"
    )


def _slot_identity(binding):
    prefix = str(binding.get("slot_prefix") or "").casefold()
    guid = str(binding.get("generator_guid") or "").casefold()
    if guid and prefix:
        return "guid", guid, prefix
    index = binding.get("generator_index")
    if index is not None and prefix:
        return "index", str(index), prefix
    return (
        "named",
        str(binding.get("generator_type") or "").casefold(),
        str(binding.get("generator_name") or "").casefold(),
        prefix,
    )


def live_export_generator_bindings(root):
    """Return Leaf Mesh/Frond slots that participate in the current export."""
    hidden_by_guid = {}
    generators = list(root.iter("Generator"))
    for generator in generators:
        guid = str(generator.findtext("GUID") or "").strip()
        if guid:
            hidden_by_guid[guid] = _truthy(
                generator.findtext("Hidden")
            )

    parent_by_guid = {}
    for link in root.iter("Link"):
        source = str(link.findtext("SourceGUID") or "").strip()
        target = str(link.findtext("TargetGUID") or "").strip()
        if source and target:
            parent_by_guid[target] = source

    export_node_count = {}
    nodes = list(root.iter("Node"))
    for node in nodes:
        guid = str(node.findtext("GeneratorGUID") or "").strip()
        if (
            not guid
            or _truthy(node.findtext("Hidden"))
            or _truthy(node.findtext("Extra/m_bDeleted"))
            or _truthy(node.findtext("Extra/m_bCulled"))
        ):
            continue
        export_node_count[guid] = export_node_count.get(guid, 0) + 1

    def effectively_hidden(guid, own_hidden):
        if own_hidden:
            return True
        seen = set()
        while guid and guid not in seen:
            seen.add(guid)
            if hidden_by_guid.get(guid):
                return True
            guid = parent_by_guid.get(guid, "")
        return False

    bindings = []
    for generator_index, generator in enumerate(generators):
        generator_type = str(generator.attrib.get("Type") or "")
        if not _is_leaf_generator(generator_type):
            continue
        generator_guid = str(
            generator.findtext("GUID") or ""
        ).strip()
        graph_visible = not effectively_hidden(
            generator_guid,
            _truthy(generator.findtext("Hidden")),
        )
        generated_node_count = int(
            export_node_count.get(generator_guid, 0)
        )
        export_participates = bool(
            graph_visible
            and (
                not nodes
                or not generator_guid
                or generated_node_count > 0
            )
        )
        properties = generator.find("Properties")
        if properties is None:
            continue
        by_name = {
            str(node.findtext("Name") or ""): node
            for node in properties.findall("Property")
        }
        for name, material_node in by_name.items():
            if not name.casefold().endswith(":material"):
                continue
            prefix = name[: -len(":Material")]
            mesh_node = next(
                (
                    node
                    for property_name, node in by_name.items()
                    if property_name.casefold()
                    == (prefix + ":Mesh").casefold()
                ),
                None,
            )
            material_id = positive_int(
                material_node.findtext("Value")
            )
            mesh_id = positive_int(
                mesh_node.findtext("Value")
                if mesh_node is not None
                else None
            )
            bindings.append({
                "generator_index": generator_index,
                "generator_guid": generator_guid,
                "generator_name": str(
                    generator.findtext("Name") or ""
                ),
                "generator_type": generator_type,
                "slot_prefix": prefix,
                "material_id": material_id,
                "mesh_id": mesh_id,
                "visible": export_participates,
                "graph_visible": graph_visible,
                "generated_node_count": generated_node_count,
                "export_participates": export_participates,
            })
    return bindings


def classify_generator_delivery(
    *,
    spm,
    connection,
    target_material_id,
    normalized_target_mesh_ids,
    live_bindings,
):
    """Classify one exact target from manifest and live export evidence."""
    connection = connection if isinstance(connection, dict) else None
    declared = [
        dict(row)
        for row in (
            connection.get("bindings") if connection is not None else []
        ) or []
        if isinstance(row, dict)
    ]
    requested = (
        connection.get("requested")
        if connection is not None
        and isinstance(connection.get("requested"), bool)
        else None
    )
    complete = (
        connection.get("complete")
        if connection is not None
        and isinstance(connection.get("complete"), bool)
        else None
    )
    policy = (
        str(connection.get("generator_variant_policy") or "")
        if connection is not None
        else ""
    )
    target_material_id = positive_int(target_material_id)
    normalized_ids = sorted({
        positive_int(value)
        for value in normalized_target_mesh_ids or []
        if positive_int(value) is not None
    })
    declared_ids = sorted({
        positive_int(row.get("target_mesh_id"))
        for row in declared
        if positive_int(row.get("target_mesh_id")) is not None
    })
    live = [
        dict(row)
        for row in live_bindings or []
        if row.get("export_participates") is True
        and positive_int(row.get("material_id"))
        == target_material_id
    ]
    live_ids = sorted({
        positive_int(row.get("mesh_id"))
        for row in live
        if positive_int(row.get("mesh_id")) is not None
    })
    result = {
        "schema_version": SCHEMA_VERSION,
        "spm": str(spm),
        "delivery_mode": DELIVERY_MODE_CONNECTION_INCOMPLETE,
        "delivery_decision": "blocked",
        "delivery_reason": "generator_connection_contract_incomplete",
        "generator_connection_requested": requested,
        "generator_connection_complete": complete,
        "generator_variant_policy": policy or None,
        "target_material_id": target_material_id,
        "normalized_target_mesh_ids": normalized_ids,
        "declared_target_mesh_ids": declared_ids,
        "live_export_participating_target_mesh_ids": live_ids,
        "generator_bindings": declared,
        "live_generator_bindings": live,
        "missing_live_bindings": [],
        "binding_mismatches": [],
        "errors": [],
    }
    if requested is False and complete is False and not declared:
        result["delivery_mode"] = (
            DELIVERY_MODE_ASSET_REGISTRATION_ONLY
        )
        result["delivery_decision"] = "pass_through"
        result["delivery_reason"] = (
            "generator_connection_not_requested"
        )
        return result
    if connection is None:
        result["errors"].append("generator_connection_missing")
        return result
    if requested is not True:
        result["errors"].append(
            "generator_connection_not_explicitly_requested"
        )
    if complete is not True:
        result["errors"].append(
            "generator_connection_not_declared_complete"
        )
    if policy != GENERATOR_VARIANT_POLICY:
        result["errors"].append(
            "generator_variant_policy_mismatch"
        )
    if not declared:
        result["errors"].append("generator_bindings_missing")
    if normalized_ids != declared_ids:
        result["errors"].append(
            "normalized_and_declared_target_mesh_sets_differ"
        )
    if normalized_ids != live_ids:
        result["errors"].append(
            "normalized_and_live_target_mesh_sets_differ"
        )

    declared_by_slot = {}
    for row in declared:
        declared_by_slot.setdefault(_slot_identity(row), []).append(row)
    live_by_slot = {}
    for row in live:
        live_by_slot.setdefault(_slot_identity(row), []).append(row)

    for slot, declared_rows in declared_by_slot.items():
        current_rows = live_by_slot.get(slot) or []
        if len(declared_rows) != 1 or len(current_rows) != 1:
            result["missing_live_bindings"].extend(
                declared_rows if not current_rows else []
            )
            result["binding_mismatches"].append({
                "slot_identity": list(slot),
                "reason": "generator_slot_cardinality_mismatch",
                "declared_count": len(declared_rows),
                "live_count": len(current_rows),
            })
            continue
        declared_row = declared_rows[0]
        current = current_rows[0]
        expected_pair = (
            positive_int(declared_row.get("target_material_id")),
            positive_int(declared_row.get("target_mesh_id")),
        )
        live_pair = (
            positive_int(current.get("material_id")),
            positive_int(current.get("mesh_id")),
        )
        if (
            expected_pair != live_pair
            or expected_pair[0] != target_material_id
        ):
            result["binding_mismatches"].append({
                "slot_identity": list(slot),
                "reason": "generator_binding_target_mismatch",
                "declared": list(expected_pair),
                "live": list(live_pair),
            })

    for slot, current_rows in live_by_slot.items():
        if len(declared_by_slot.get(slot) or []) != 1:
            result["binding_mismatches"].append({
                "slot_identity": list(slot),
                "reason": "live_generator_slot_not_declared_exactly_once",
                "declared_count": len(
                    declared_by_slot.get(slot) or []
                ),
                "live_count": len(current_rows),
            })

    if result["missing_live_bindings"]:
        result["errors"].append("visible_generator_slot_missing")
    if result["binding_mismatches"]:
        result["errors"].append("generator_binding_mismatch")
    result["errors"] = sorted(set(result["errors"]))
    if declared and not result["errors"]:
        result["delivery_mode"] = DELIVERY_MODE_RENDER_CONNECTED
        result["delivery_decision"] = "normalize_part"
        result["delivery_reason"] = (
            "generator_connection_matches_live_export"
        )
    return result
