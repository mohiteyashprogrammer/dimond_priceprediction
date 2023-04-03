from setuptools import setup,find_packages
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requiremenst(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [i.replace("\n","") for i in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements



setup(
    name='Regressionproject',
    version='0.0.1',
    author="yash mohite",
    author_email="mohite.yassh@gmail.com",
    packages=find_packages(),
    install_requires = get_requiremenst("requirements.txt")
    
)