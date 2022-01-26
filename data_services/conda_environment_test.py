import platform
import yaml
import pkg_resources
import re
import logging

log = logging.getLogger(__name__)


def convert_conda_yaml_to_requirement(conda_array) :
    '''
    Convert the conda.yaml syntax to requirements.txt syntax :
    for now : 
       - select "dependencies" key
       - transform = into ==
       - add pip packages dependencies to the list of other dependencies
    Additionally remove python requirement (not supported by pkg_resources.require)
    Also need to remove pip -e "install"
    '''
    # get dependencies
    dep_array = [v for v in conda_array["dependencies"] if type(v) == str] 
    pip_require = [v for v in conda_array["dependencies"]
                   if type(v) == dict and "pip" in v.keys()][0]["pip"]
    # remove " -e  " install type :
    pip_require = [v for v in pip_require if (re.match(r"^ *-e ",v) == None)]
    
    # need to add extra = if no < or >
    dep_array_conv = [x.replace('=','==') for x in dep_array]
    dep_array_conv = [x.replace(r'>==','>=').replace('<==','<=').replace('===','==')
                      for x in dep_array_conv]
    # put back pip requirement in place
    # assumes it is at the end
    dep_array_conv = dep_array_conv + pip_require
    # remove python version check
    dep_array_conv = [x for x in dep_array_conv if re.match('^python[<.>,=]=',x) == None]
    return dep_array_conv



def conda_python_version_requirement(conda_array):
    '''
    Return the python version required if present in the conda.yaml
    Otherwise return None
    
    '''
    # get dependencies
    dep_array = [v for v in conda_array["dependencies"] if type(v) == str] 
    # get Python version
    python_req =  [x for x in dep_array if re.match('^python[<.>,=]',x) != None]
    if len(python_req) == 0 :
        return None
    else :
        # Only return 1st occurence
        return python_req[0].replace('python','')



def check_python(requirement, value) :
    '''
    Check if a Python version abide by a Python version requirement
    WARNING :
    this can only check 1 condition, can not check multiple conditions
    separated by ,
    '''
    condition = re.findall('[<,>,=]=*', requirement)[0]
    condition = condition.replace('=','==')
    condition = condition.replace('<==','<=').replace('>==','>=').replace('===','==')
    version_req = re.findall('[0-9.]+', requirement)[0]
    len_version = len(version_req.split('.'))
    value = ".".join(value.split('.')[0:len_version])
    value = pkg_resources.parse_version(value)
    version_req = pkg_resources.parse_version(version_req)
    test = eval("value "+condition+" version_req")
    return test


    


def check_environment(filename = 'conda.yaml') :
    '''
    Check that the current conda environment abide by the filename (conda.yaml)
    and raise an error if not.
    A good place to put the function is in the file ./src/{project_name}/pipeline.py
    at the beginning of the create_pipelines function
    '''
    with open(filename) as stream :
        values = yaml.safe_load(stream)
    pkg_req = convert_conda_yaml_to_requirement(values)
    pkg_resources.require(pkg_req)
    python_req = conda_python_version_requirement(values)
    if (python_req != None) :
        python_ver = platform.python_version()
        if not(check_python(python_req, python_ver)) :
            raise(Exception(f"python version {python_ver} is not compatible "
                            f"with conda.yaml python requirement {python_req}"))
    log.info(f"Conda environment matches the requirements of {filename}")


if __name__  == "__main__" :
    check_environment()
    

