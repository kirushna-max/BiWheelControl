#!/usr/bin/env python3
"""Convert the repository's Bipedal URDF to an MJCF model.

Running this file without arguments converts ``Bipedal/urdf/Bipedal.urdf`` to
``Bipedal/urdf/Bipedal.xml``.  Input and output paths can still be supplied
when converting another model.
"""

import argparse
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def vec(text, default="0 0 0"):
    return (text or default).split()


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def main():
    project_root = Path(__file__).resolve().parent
    default_urdf = project_root / "Bipedal" / "urdf" / "Bipedal.urdf"
    default_output = default_urdf.with_suffix(".xml")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "urdf",
        nargs="?",
        type=Path,
        default=default_urdf,
        help=f"URDF to convert (default: {default_urdf})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_output,
        help=f"MJCF output path (default: {default_output})",
    )
    args = parser.parse_args()

    urdf = args.urdf.resolve()
    output = args.output.resolve()
    if not urdf.is_file():
        parser.error(f"URDF not found: {urdf}")

    root = ET.parse(urdf).getroot()
    links = {x.attrib["name"]: x for x in root.findall("link")}
    joints = {x.find("child").attrib["link"]: x for x in root.findall("joint")}
    child_names = set(joints)
    roots = [name for name in links if name not in child_names]
    robot_root = roots[0]

    mj = ET.Element("mujoco", model=root.attrib.get("name", "robot"))
    ET.SubElement(mj, "compiler", meshdir=".", angle="radian")
    ET.SubElement(mj, "option", gravity="0 0 -9.81", timestep="0.002")
    asset = ET.SubElement(mj, "asset")
    mesh_names = {}

    def mesh_file(filename):
        filename = filename.replace("package://Bipedal/", "")
        filename = filename.replace("package://", "")
        candidate = (urdf.parent.parent / filename).resolve()
        if not candidate.exists():
            candidate = (urdf.parent / filename).resolve()
        return candidate

    for link_name, link in links.items():
        visual = link.find("visual")
        if visual is None:
            continue
        mesh = visual.find("geometry/mesh")
        if mesh is not None:
            source = mesh_file(mesh.attrib["filename"])
            if not source.is_file():
                raise FileNotFoundError(
                    f"Mesh referenced by {urdf.name} was not found: {source}"
                )
            name = safe_name(link_name)
            mesh_names[link_name] = name
            # Paths are relative to the generated MJCF file.
            rel = Path("..") / source.relative_to(urdf.parent.parent)
            ET.SubElement(asset, "mesh", name=name, file=rel.as_posix())

    worldbody = ET.SubElement(mj, "worldbody")
    ET.SubElement(worldbody, "geom", name="ground", type="plane", size="5 5 .1", rgba=".8 .8 .8 1")

    def add_link(parent, link_name, is_root=False):
        link = links[link_name]
        joint = joints.get(link_name)
        body_attrs = {"name": safe_name(link_name)}
        if joint is not None:
            origin = joint.find("origin")
            if origin is not None:
                body_attrs["pos"] = " ".join(vec(origin.attrib.get("xyz")))
                body_attrs["euler"] = " ".join(vec(origin.attrib.get("rpy")))
        body = ET.SubElement(parent, "body", **body_attrs)
        if is_root:
            ET.SubElement(body, "freejoint", name="root_freejoint")
        elif joint is not None and joint.attrib.get("type") not in ("fixed", ""):
            axis = joint.find("axis")
            attrs = {"name": safe_name(joint.attrib["name"]), "type": "hinge",
                     "axis": " ".join(vec(axis.attrib.get("xyz") if axis is not None else None))}
            limit = joint.find("limit")
            if limit is not None and float(limit.attrib.get("lower", "0")) < float(limit.attrib.get("upper", "0")):
                attrs["range"] = f'{limit.attrib["lower"]} {limit.attrib["upper"]}'
                attrs["limited"] = "true"
            ET.SubElement(body, "joint", **attrs)

        inertial = link.find("inertial")
        if inertial is not None:
            origin = inertial.find("origin")
            mass = inertial.find("mass")
            ine = inertial.find("inertia")
            attrs = {"mass": mass.attrib["value"] if mass is not None else "1"}
            if origin is not None:
                attrs["pos"] = " ".join(vec(origin.attrib.get("xyz")))
            if ine is not None:
                attrs["diaginertia"] = " ".join(ine.attrib.get(k, "0") for k in ("ixx", "iyy", "izz"))
            ET.SubElement(body, "inertial", **attrs)

        if link_name in mesh_names:
            visual = link.find("visual")
            origin = visual.find("origin")
            attrs = {"name": safe_name(link_name) + "_visual", "type": "mesh", "mesh": mesh_names[link_name], "contype": "0", "conaffinity": "0"}
            if origin is not None:
                attrs["pos"] = " ".join(vec(origin.attrib.get("xyz")))
                attrs["euler"] = " ".join(vec(origin.attrib.get("rpy")))
            color = visual.find("material/color")
            if color is not None:
                attrs["rgba"] = color.attrib.get("rgba", "0.8 0.8 0.8 1")
            ET.SubElement(body, "geom", **attrs)
        for child, child_joint in joints.items():
            if child_joint.find("parent").attrib["link"] == link_name:
                add_link(body, child)

    add_link(worldbody, robot_root, True)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(mj).write(output, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
