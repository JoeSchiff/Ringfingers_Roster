
# Desc: Convert cython extention class properties to newer syntax

# Works only with spaces, not tabs


"""
todo:
keep cdef class ?
docstring after @property?
indent testing
detect when indent nec, not always
iter files +
"""


"""
problem files:
enum
"""


from glob import glob
import os



pyav_path = r"C:\Users\jschiffler\Desktop\pyav\PyAV-main"





def remove_indent(line):
    if remove_indent_b:
        return line.replace(" "*4, "", 1)
    else:
        return line

def get_indent_amount(line):
    return len(line) - len(line.lstrip())

def get_indent():
    return " " * indent_amount


def one_line_convert(line, orig_name):
    """For `__get__` and the second line of `__set__` and `__del__`"""
    global remove_indent_b
    remove_indent_b = True
    new_line = line.replace(orig_name, property_name)
    return remove_indent(new_line)

def two_line_convert(line, orig_name, new_name):
    """For `__set__` and `__del__`"""
    first_line = f"{get_indent()}@{property_name}.{new_name}"
    second_line = one_line_convert(line, orig_name)
    return "\n".join((first_line, second_line))



for file_path in glob(pyav_path + "\**\*.pyx", recursive=True):
    end_s = ""
    remove_indent_b = False
    file_name = os.path.basename(file_path)
    with open(file_path) as file:
        print("=============", file_name)
        for line in file.read().splitlines():
            new_line = line
            # Class name
            if line.strip().startswith("cdef class "):
                new_line = line.replace("cdef ", "@cython.cclass\n")  ##
            # Init prop
            elif line.strip().startswith("property "):
                remove_indent_b = True
                indent_amount = get_indent_amount(line)
                property_name = line.split("    property ")[1].split(":")[0]
                new_line = f"{get_indent()}@property"
            # Prop get
            elif line.strip().startswith("def __get__(self"):
                new_line = one_line_convert(line, "__get__")
            # Prop set
            elif line.strip().startswith("def __set__(self"):
                new_line = two_line_convert(line, "__set__", "setter")
            # Prop del
            elif line.strip().startswith("def __del__(self"):
                new_line = two_line_convert(line, "__del__", "deleter")
            # Already correct
            elif line.strip().startswith("@property"):
                remove_indent_b = False
            else:
                new_line = remove_indent(new_line)  ##
            end_s = "\n".join((end_s, new_line))
    print(end_s)
    input()








