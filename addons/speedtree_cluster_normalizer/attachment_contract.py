import copy
import gzip
import hashlib
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import bpy
from mathutils import Vector


XML_ATTR_RE = re.compile(r'([A-Za-z][A-Za-z0-9]*)="([^"]*)"')
XML_SOURCE_RE = re.compile(r'<SpeedTreeRaw\b[^>]*\bSource="([^"]+)"')
XML_SCALE_CANDIDATES = (100.0, 1.0, 3.28084, 30.48, 0.01)
GEOMETRY_SUPPORTED_ATTACHMENT_POLICY = "geometry_supported_xml_root_segment"
SPM_STRUCTURAL_SEMANTIC_PROJECTION_VERSION = 1

_IGNORED_SPM_SUBTREE_TAGS = frozenset({"Thumbnail", "Preview"})
_SPM_MATERIAL_GEOMETRY_TAGS = frozenset(
    {
        "CutoutMeshID",
        "SupplementalCutoutMeshIDs",
        "UVAreas",
        "Width",
        "Height",
        "UnwrapScale",
        "AtlasMaker",
    }
)


def _parse_speedtree_xml_number(value, label):
    """Parse SpeedTree Raw XML numbers written with dot or comma decimals."""
    text = str("" if value is None else value).strip()
    if not text:
        raise ValueError(f"{label} is empty")
    if "," in text:
        if "." in text or text.count(",") != 1:
            raise ValueError(
                f"{label} has ambiguous decimal separators: {text!r}"
            )
        text = text.replace(",", ".")
    number = float(text)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {value!r}")
    return number


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_xml_tag(tag):
    return str(tag).rsplit("}", 1)[-1]


def _is_spm_material_slot_property(name):
    folded = str(name or "").strip().casefold()
    return (
        folded.endswith(":material")
        or folded.startswith("materials:")
        or folded
        in {
            "material:frond",
            "mesh:material",
            "mesh:render material",
        }
    )


def _spm_material_slot_assignment_state(raw_value):
    value = str(raw_value or "").strip()
    if not value or value.casefold() in {"none", "null", "unassigned"}:
        return "UNASSIGNED"
    try:
        if float(value) < 0:
            return "UNASSIGNED"
    except ValueError:
        pass
    return "ASSIGNED"


def _remove_non_structural_spm_content(parent):
    """Project an SPM onto structure that can affect normalized delivery."""
    for child in list(parent):
        tag = _local_xml_tag(child.tag)
        remove = tag in _IGNORED_SPM_SUBTREE_TAGS
        if tag == "Material_v8":
            cutout = child.findtext("CutoutMeshID")
            supplemental = child.find("SupplementalCutoutMeshIDs")
            uv_areas = child.find("UVAreas")
            has_geometry = (
                str(cutout or "").strip() not in {"", "-1"}
                or (
                    supplemental is not None
                    and str(supplemental.get("Count") or "0") != "0"
                )
                or (
                    uv_areas is not None
                    and str(uv_areas.get("Count") or "0") != "0"
                )
            )
            if not has_geometry:
                parent.remove(child)
                continue
            child.attrib.pop("Name", None)
            for material_child in list(child):
                if (
                    _local_xml_tag(material_child.tag)
                    not in _SPM_MATERIAL_GEOMETRY_TAGS
                ):
                    child.remove(material_child)
            continue
        if tag == "Property":
            name = str(child.findtext("Name") or "").strip().casefold()
            remove = name.startswith("vertex color:")
            if not remove and _is_spm_material_slot_property(name):
                value = child.find("Value")
                if value is not None:
                    value.text = _spm_material_slot_assignment_state(value.text)
        if remove:
            parent.remove(child)
            continue
        _remove_non_structural_spm_content(child)
        if (
            _local_xml_tag(child.tag) == "Assets"
            and not list(child)
            and not str(child.text or "").strip()
        ):
            parent.remove(child)


def _decode_spm_xml(path):
    payload = Path(path).read_bytes()
    if payload.startswith(b"\x1f\x8b"):
        payload = gzip.decompress(payload)
    return payload.decode("utf-8")


def spm_structural_semantic_fingerprint(path):
    """Hash SPM bone/geometry/cutout/generator semantics, not texture authoring."""
    root = ET.fromstring(_decode_spm_xml(path))
    projected = copy.deepcopy(root)
    _remove_non_structural_spm_content(projected)
    source = ET.tostring(
        projected,
        encoding="unicode",
        short_empty_elements=True,
    )
    source = re.sub(r">\s+<", "><", source).strip()
    payload = json.dumps(
        {
            "projection_version": SPM_STRUCTURAL_SEMANTIC_PROJECTION_VERSION,
            "source": source,
            "context": {},
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _resolved_existing_path(value, label, *, relative_to=None):
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"{label} is required.")
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        path = candidate.resolve()
    elif relative_to is not None:
        path = (Path(relative_to).resolve() / candidate).resolve()
    else:
        path = Path(bpy.path.abspath(raw)).resolve()
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path


def resolve_source_xml_path(scene, explicit_path):
    explicit = str(explicit_path or "").strip()
    if explicit:
        return _resolved_existing_path(explicit, "Source 3D XML")
    repair_settings = getattr(scene, "speedtree_bwr_settings", None)
    repair_xml = str(getattr(repair_settings, "xml_path", "") or "").strip()
    if repair_xml:
        return _resolved_existing_path(repair_xml, "Bone Repair Source XML")
    raise ValueError(
        "Source 3D XML is required. Set it explicitly or load it through "
        "SpeedTree Bone Weight Repair first."
    )


def _parse_speedtree_xml(path):
    bones = []
    source_spm = ""
    with open(path, "r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, 1):
            if not source_spm:
                match = XML_SOURCE_RE.search(line)
                if match is not None:
                    source_spm = match.group(1)
            if "<Bone " not in line:
                continue
            for chunk in line.split("<Bone ")[1:]:
                close = chunk.find(">")
                if close < 0:
                    raise ValueError(
                        f"SpeedTree Raw XML has an unterminated Bone entry at "
                        f"{path}:{line_number}."
                    )
                attrs = dict(XML_ATTR_RE.findall(chunk[: close if close >= 0 else None]))
                try:
                    required = (
                        "ID",
                        "ParentID",
                        "StartX",
                        "StartY",
                        "StartZ",
                        "EndX",
                        "EndY",
                        "EndZ",
                    )
                    missing = [key for key in required if key not in attrs]
                    if missing:
                        raise KeyError(", ".join(missing))
                    bones.append(
                        {
                            "id": int(attrs["ID"]),
                            "parent_id": int(attrs["ParentID"]),
                            "start_raw": Vector(
                                (
                                    _parse_speedtree_xml_number(
                                        attrs["StartX"], "StartX"
                                    ),
                                    _parse_speedtree_xml_number(
                                        attrs["StartY"], "StartY"
                                    ),
                                    _parse_speedtree_xml_number(
                                        attrs["StartZ"], "StartZ"
                                    ),
                                )
                            ),
                            "end_raw": Vector(
                                (
                                    _parse_speedtree_xml_number(
                                        attrs["EndX"], "EndX"
                                    ),
                                    _parse_speedtree_xml_number(
                                        attrs["EndY"], "EndY"
                                    ),
                                    _parse_speedtree_xml_number(
                                        attrs["EndZ"], "EndZ"
                                    ),
                                )
                            ),
                            "radius_raw": _parse_speedtree_xml_number(
                                attrs.get("Radius", "0") or 0.0,
                                "Radius",
                            ),
                            "generator": attrs.get("Generator", ""),
                        }
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"SpeedTree Raw XML has an invalid Bone entry at "
                        f"{path}:{line_number}: {exc}"
                    ) from exc
    if not source_spm:
        raise ValueError(f"SpeedTree Raw XML lacks its Source SPM: {path}")
    if not bones:
        raise ValueError(f"SpeedTree Raw XML contains no usable Bone entries: {path}")
    ids = [bone["id"] for bone in bones]
    if len(ids) != len(set(ids)):
        raise ValueError(f"SpeedTree Raw XML contains duplicate Bone IDs: {path}")
    invalid_parent_ids = sorted(
        {bone["parent_id"] for bone in bones if bone["parent_id"] < -1}
    )
    if invalid_parent_ids:
        raise ValueError(
            "SpeedTree Raw XML contains invalid negative ParentID values; only -1 "
            f"is a structural root: {invalid_parent_ids}"
        )
    return source_spm, bones


def _median(values):
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) * 0.5


def _choose_xml_scale(bones, armature):
    heads = [armature.matrix_world @ bone.head_local for bone in armature.data.bones]
    if not heads:
        raise ValueError("Source armature has no joints for XML scale validation.")
    scores = []
    for scale in XML_SCALE_CANDIDATES:
        endpoints = [
            point / scale
            for bone in bones
            for point in (bone["start_raw"], bone["end_raw"])
        ]
        distances = [min((head - point).length for point in endpoints) for head in heads]
        scores.append(
            {
                "scale": float(scale),
                "median_nearest": float(_median(distances)),
                "max_nearest": float(max(distances)),
            }
        )
    scores.sort(key=lambda row: (row["median_nearest"], row["max_nearest"], row["scale"]))
    return scores[0]["scale"], scores


def _path_key(path):
    return os.path.normcase(os.path.normpath(str(Path(path).resolve())))


def _asset_stem_key(value):
    stem = str(value or "").casefold()
    return stem[3:] if stem.startswith("sk_") else stem


def load_attachment_contract(scene, source, armature, explicit_xml_path=""):
    xml_path = resolve_source_xml_path(scene, explicit_xml_path)
    source_spm_text, bones = _parse_speedtree_xml(xml_path)
    source_spm = _resolved_existing_path(
        source_spm_text,
        "XML Source SPM",
        relative_to=xml_path.parent,
    )
    source_fbx = _resolved_existing_path(
        source.get("codex_source_fbx", ""),
        "Source mesh FBX provenance",
    )
    repair_settings = getattr(scene, "speedtree_bwr_settings", None)
    repair_spm_text = str(getattr(repair_settings, "spm_path", "") or "").strip()
    if repair_spm_text:
        repair_spm = _resolved_existing_path(repair_spm_text, "Bone Repair Source SPM")
        if _path_key(repair_spm) != _path_key(source_spm):
            raise ValueError(
                "Source 3D XML points to a different SPM than Bone Repair: "
                f"{source_spm} vs {repair_spm}"
            )
    stems = {
        _asset_stem_key(xml_path.stem),
        _asset_stem_key(source_spm.stem),
        _asset_stem_key(source_fbx.stem),
    }
    if len(stems) != 1:
        raise ValueError(
            "Source XML/SPM/FBX stems do not identify the same 3D cluster: "
            f"{xml_path.stem}, {source_spm.stem}, {source_fbx.stem}"
        )
    xml_mtime = xml_path.stat().st_mtime_ns
    stale_inputs = [
        path for path in (source_spm, source_fbx) if path.stat().st_mtime_ns > xml_mtime
    ]
    if stale_inputs:
        raise ValueError(
            "Source 3D XML is older than its source export; regenerate XML before "
            "normalization: " + ", ".join(str(path) for path in stale_inputs)
        )
    scale, scale_scores = _choose_xml_scale(bones, armature)
    world_bones = []
    for bone in bones:
        world_bones.append(
            {
                "id": int(bone["id"]),
                "parent_id": int(bone["parent_id"]),
                "start_world": bone["start_raw"] / scale,
                "end_world": bone["end_raw"] / scale,
                "radius_world": float(bone["radius_raw"] / scale),
                "generator": bone["generator"],
            }
        )
    roots = [bone for bone in world_bones if bone["parent_id"] == -1]
    if not roots:
        raise ValueError("Source 3D XML contains no structural root Bone (ParentID=-1).")
    return {
        "xml_path": str(xml_path),
        "xml_sha256": _sha256(xml_path),
        "xml_mtime_ns": int(xml_mtime),
        "source_spm": str(source_spm),
        "source_spm_sha256": _sha256(source_spm),
        "source_spm_semantic_projection_version":
            SPM_STRUCTURAL_SEMANTIC_PROJECTION_VERSION,
        "source_spm_semantic_fingerprint":
            spm_structural_semantic_fingerprint(source_spm),
        "source_fbx": str(source_fbx),
        "source_fbx_sha256": _sha256(source_fbx),
        "scale": float(scale),
        "scale_scores": scale_scores,
        "bones": world_bones,
        "roots": roots,
    }


def _bone_world_head(armature, bone):
    return armature.matrix_world @ bone.head_local


def match_root_attachment(
    contract,
    armature,
    representative_bone,
    endpoint_bone,
    geometry_scale,
    used_root_ids,
):
    name = representative_bone.name
    head = _bone_world_head(armature, representative_bone)
    endpoint = _bone_world_head(armature, endpoint_bone) if endpoint_bone else None
    roots = [bone for bone in contract["roots"] if bone["id"] not in used_root_ids]
    if not roots:
        raise ValueError(f"No unused XML structural root remains for {name}.")
    lowered = name.casefold()
    candidates = []
    if lowered.endswith("_start"):
        for root in roots:
            start_error = float((head - root["start_world"]).length)
            end_error = (
                float((endpoint - root["end_world"]).length)
                if endpoint is not None
                else 0.0
            )
            candidates.append((start_error + end_error, start_error, end_error, root))
        policy = "xml_root_start_to_start_joint"
    elif lowered.endswith("_end") and representative_bone.parent is None:
        for root in roots:
            end_error = float((head - root["end_world"]).length)
            candidates.append((end_error, 0.0, end_error, root))
        policy = "xml_root_end_identifies_missing_start_joint"
    else:
        raise ValueError(
            f"Prototype representative '{name}' is not an XML structural Start root "
            "or an orphan End with a missing Start joint."
        )
    candidates.sort(key=lambda row: (row[0], row[3]["id"]))
    best = candidates[0]
    tolerance = max(float(geometry_scale) * 1.0e-4, 1.0e-6)
    if best[1] > tolerance or best[2] > tolerance:
        raise ValueError(
            f"XML root match for '{name}' exceeds relative tolerance: "
            f"start={best[1]:.9g}, end={best[2]:.9g}, tolerance={tolerance:.9g}."
        )
    if len(candidates) > 1 and math.isclose(
        candidates[1][0], best[0], rel_tol=0.0, abs_tol=tolerance * 1.0e-3
    ):
        raise ValueError(f"XML structural root match is ambiguous for '{name}'.")
    root = best[3]
    used_root_ids.add(root["id"])
    direction = root["end_world"] - root["start_world"]
    if direction.length <= tolerance:
        raise ValueError(f"XML structural root segment is degenerate for '{name}'.")
    return {
        "xml_bone_id": int(root["id"]),
        "xml_parent_id": int(root["parent_id"]),
        "xml_generator": root["generator"],
        "xml_start_world": root["start_world"].copy(),
        "xml_end_world": root["end_world"].copy(),
        "xml_radius_world": float(root["radius_world"]),
        "representative_bone": name,
        "endpoint_bone": endpoint_bone.name if endpoint_bone else "",
        "match_policy": policy,
        "start_match_error": float(best[1]),
        "end_match_error": float(best[2]),
        "match_tolerance": float(tolerance),
    }


def fit_attachment_to_geometry(attachment, world_points, geometry_scale):
    """Resolve the usable root span from the structural XML segment and geometry."""
    points = [Vector(point) for point in world_points]
    scale = float(geometry_scale)
    if len(points) < 3 or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Geometry-supported attachment requires measurable geometry.")

    start = Vector(attachment["xml_start_world"])
    end = Vector(attachment["xml_end_world"])
    segment = end - start
    segment_length = float(segment.length)
    tolerance = max(scale * 1.0e-4, 1.0e-6)
    if segment_length <= tolerance:
        raise ValueError("Geometry-supported XML root segment is degenerate.")
    axis = segment / segment_length
    projections = [float((point - start).dot(axis)) for point in points]
    minimum_projection = min(projections)
    maximum_projection = max(projections)
    if (
        maximum_projection < -tolerance
        or minimum_projection > segment_length + tolerance
    ):
        raise ValueError(
            "Prototype geometry does not overlap its XML structural root segment."
        )

    support_distance = min(
        max(minimum_projection, 0.0),
        segment_length,
    )
    if support_distance <= tolerance:
        support_distance = 0.0
    remaining_length = segment_length - support_distance
    direction_length = min(remaining_length, scale)
    if direction_length <= tolerance:
        raise ValueError(
            "Prototype geometry leaves no usable attachment direction on its "
            "XML structural root segment."
        )

    effective_start = start + axis * support_distance
    effective_end = effective_start + axis * direction_length
    result = dict(attachment)
    result.update(
        {
            "effective_attachment_world": effective_start,
            "effective_endpoint_world": effective_end,
            "effective_support_distance_world": float(support_distance),
            "effective_direction_length_world": float(direction_length),
            "effective_geometry_scale_world": scale,
            "effective_support_min_projection_world": float(minimum_projection),
            "effective_support_max_projection_world": float(maximum_projection),
            "effective_support_tolerance_world": float(tolerance),
            "effective_attachment_policy": GEOMETRY_SUPPORTED_ATTACHMENT_POLICY,
        }
    )
    return result


def attachment_origin_world(attachment):
    return Vector(
        attachment.get("effective_attachment_world")
        or attachment["xml_start_world"]
    )


def attachment_endpoint_world(attachment):
    return Vector(
        attachment.get("effective_endpoint_world")
        or attachment["xml_end_world"]
    )


def serialized_contract_source(contract):
    return {
        "xml_path": contract["xml_path"],
        "xml_sha256": contract["xml_sha256"],
        "xml_mtime_ns": contract["xml_mtime_ns"],
        "source_spm": contract["source_spm"],
        "source_spm_sha256": contract["source_spm_sha256"],
        "source_spm_semantic_projection_version": contract[
            "source_spm_semantic_projection_version"
        ],
        "source_spm_semantic_fingerprint": contract[
            "source_spm_semantic_fingerprint"
        ],
        "source_fbx": contract["source_fbx"],
        "source_fbx_sha256": contract["source_fbx_sha256"],
        "scale": contract["scale"],
        "scale_scores": contract["scale_scores"],
        "root_ids": [int(bone["id"]) for bone in contract["roots"]],
    }


def serialized_attachment(attachment):
    vector_keys = {
        "xml_start_world",
        "xml_end_world",
        "effective_attachment_world",
        "effective_endpoint_world",
    }
    return {
        key: (
            [float(value) for value in attachment[key]]
            if key in vector_keys
            else attachment[key]
        )
        for key in (
            "xml_bone_id",
            "xml_parent_id",
            "xml_generator",
            "xml_start_world",
            "xml_end_world",
            "xml_radius_world",
            "representative_bone",
            "endpoint_bone",
            "match_policy",
            "start_match_error",
            "end_match_error",
            "match_tolerance",
            "effective_attachment_world",
            "effective_endpoint_world",
            "effective_support_distance_world",
            "effective_direction_length_world",
            "effective_geometry_scale_world",
            "effective_support_min_projection_world",
            "effective_support_max_projection_world",
            "effective_support_tolerance_world",
            "effective_attachment_policy",
        )
        if key in attachment
    }
