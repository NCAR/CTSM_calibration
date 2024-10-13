import xml.etree.ElementTree as ET

def prettify(element, indent='  ', level=0):
    """Add indentation and newlines to an XML element and all of its children."""
    i = "\n" + level * indent
    if len(element):
        if not element.text or not element.text.strip():
            element.text = i + indent
        if not element.tail or not element.tail.strip():
            element.tail = i
        for element in element:
            prettify(element, indent, level+1)
        if not element.tail or not element.tail.strip():
            element.tail = i
    else:
        if level and (not element.tail or not element.tail.strip()):
            element.tail = i

def insert_cpu_bind(file_path,cpulist):
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Find the specific mpirun element
    mpirun = root.find(".//mpirun[@mpilib='default']")
    if mpirun is None:
        print("mpirun element with mpilib='default' not found.")
        return

    # Navigate to the arguments section
    arguments = mpirun.find("arguments")
    if arguments is None:
        print("<arguments> element not found within <mpirun>.")
        return

    # Check if the cpu_bind arg already exists
    cpu_bind_arg = arguments.find(".//arg[@name='cpu_bind']")
    if cpu_bind_arg is not None:
        print("cpu_bind arg already exists.")
        return

    # Define the new cpu_bind arg element and insert it
    new_cpu_bind_arg = ET.Element('arg', {'name': 'cpu_bind'})
    new_cpu_bind_arg.text = f' --cpu-bind list:{cpulist}'
    arguments.append(new_cpu_bind_arg)

    # Add indentation to make the XML pretty
    prettify(root)

    # Save the modified XML back to the file
    tree.write(file_path, encoding='utf-8', xml_declaration=True)
    print(f"cpu_bind {cpulist} arg inserted successfully.")

# # Path to the XML file
# xml_file_path = 'env_mach_specific.xml'
# insert_cpu_bind(xml_file_path,'10:12')
