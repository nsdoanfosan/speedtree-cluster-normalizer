"""Smoke-test finalized physical capture map list normalization."""

import addon_utils


def main():
    addon_utils.enable(
        "speedtree_cluster_normalizer",
        default_set=False,
        persistent=False,
    )
    from speedtree_cluster_normalizer.cluster_handoff import (
        capture_maps_by_role,
    )

    rows = [
        {"role": "Color", "path": "color.tga"},
        {"role": "Opacity", "path": "opacity.tga"},
    ]
    mapped = capture_maps_by_role(rows)
    if mapped["Color"] is not rows[0] or mapped["Opacity"] is not rows[1]:
        raise RuntimeError("Capture map list was not normalized by role")
    if capture_maps_by_role(mapped) != mapped:
        raise RuntimeError("Role-keyed capture map input was not preserved")
    try:
        capture_maps_by_role(
            [{"role": "Color"}, {"role": "Color"}]
        )
    except ValueError:
        pass
    else:
        raise RuntimeError("Duplicate capture map roles were accepted")


if __name__ == "__main__":
    main()
