import os
import xml.etree.ElementTree as ET
import json
from xml.dom import minidom

def parse_density_xml(file_path, target_set_id=None):
    """
    Parses the cfsErsatzMaterial XML file to extract header metadata 
    and shapeParamElement feature placements and sizes for a specific set.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Extract metadata from the header
    metadata = {}
    header = root.find('header')
    if header is not None:
        for child in header:
            tag_data = dict(child.attrib)
            # Retain nested element attributes if present
            if len(child) > 0:
                tag_data['nested_elements'] = {sub.tag: dict(sub.attrib) for sub in child}
            metadata[child.tag] = tag_data

    # Locate the relevant set
    sets = root.findall('set')
    if not sets:
        return metadata, {}

    target_set = None
    if target_set_id is not None:
        for s in sets:
            if s.attrib.get('id') == str(target_set_id):
                target_set = s
                break
    else:
        # Default to the last set if no ID is specified
        target_set = sets[-1]

    if target_set is None:
        raise ValueError(f"Set ID {target_set_id} not found.")

    # Extract feature placement and sizes from shapeParamElement tags
    features = {}
    for shape_elem in target_set.findall('shapeParamElement'):
        shape_id = shape_elem.attrib.get('shape')
        if shape_id not in features:
            features[shape_id] = {'start_node': {}, 'end_node': {}, 'profile_size': None}
        
        elem_type = shape_elem.attrib.get('type')
        design_val = float(shape_elem.attrib.get('design', 0.0))
        
        if elem_type == 'node':
            dof = shape_elem.attrib.get('dof')
            tip = shape_elem.attrib.get('tip')
            if tip == 'start':
                features[shape_id]['start_node'][dof] = design_val
            elif tip == 'end':
                features[shape_id]['end_node'][dof] = design_val
        elif elem_type == 'profile':
            features[shape_id]['profile_size'] = design_val

    return metadata, features

def append_density_xml(s, density_field, grid_shape, output_path, transition_val, extension_val, iteration_id):
    """
    Appends a new <set> to an existing cfsErsatzMaterial XML file.
    If the file does not exist, it initializes the root and header.
    """
    nx, ny = grid_shape
    
    internal_transition = transition_val / 2.0
    external_transition = internal_transition + extension_val

    if os.path.exists(output_path):
        tree = ET.parse(output_path)
        root = tree.getroot()
    else:
        root = ET.Element("cfsErsatzMaterial")
        header = ET.SubElement(root, "header")
        ET.SubElement(header, "mesh", x=str(nx), y=str(ny), z="1")
        ET.SubElement(header, "featureMapping", 
                      InternalTransition=str(internal_transition), 
                      ExternalTransition=str(external_transition))
        tree = ET.ElementTree(root)
    
    set_elem = ET.SubElement(root, "set", id=str(iteration_id))
    
    for i, val in enumerate(density_field):
        ET.SubElement(set_elem, "element", nr=str(i+1), type="density", design=str(val), physical=str(val))
        
    num_features = len(s) // 5
    nr = 0
    for shape_idx in range(num_features):
        px, py, qx, qy, r = s[shape_idx*5 : shape_idx*5+5]
        ET.SubElement(set_elem, "shapeParamElement", nr=str(nr), type="node", dof="x", tip="start", shape=str(shape_idx), design=str(px))
        nr += 1
        ET.SubElement(set_elem, "shapeParamElement", nr=str(nr), type="node", dof="y", tip="start", shape=str(shape_idx), design=str(py))
        nr += 1
        ET.SubElement(set_elem, "shapeParamElement", nr=str(nr), type="node", dof="x", tip="end", shape=str(shape_idx), design=str(qx))
        nr += 1
        ET.SubElement(set_elem, "shapeParamElement", nr=str(nr), type="node", dof="y", tip="end", shape=str(shape_idx), design=str(qy))
        nr += 1
        ET.SubElement(set_elem, "shapeParamElement", nr=str(nr), type="profile", shape=str(shape_idx), design=str(r))
        nr += 1

    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ", level=0)

    tree.write(output_path, encoding="utf-8", xml_declaration=True)

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    file_name = os.path.join(script_dir, "..", "densityfiles", "truncated.density.xml")
    meta, feats = parse_density_xml(file_name)
    
    print("--- Metadata ---")
    print(json.dumps(meta, indent=2))
    print("\n--- Features (Last Set) ---")
    print(json.dumps(feats, indent=2))