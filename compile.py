import glob
import os
from shutil import copyfile

from numpy import save

def nuitka_compile():
    pyfiles = glob.glob("face_recognition_sdk/**/*.py", recursive=True)
    pyfiles.extend(glob.glob("api/**/*.py", recursive=True))
    pyfiles.extend(glob.glob("core/**/*.py", recursive=True))
    pyfiles.extend(["insight_face.py", "share_param.py"])

    for name in pyfiles:
        savepath = os.path.join("compile", os.path.dirname(name))
        cmd = f"python3 -m nuitka --module {name} --output-dir={savepath} --remove-output --no-pyi-file"
        os.system(cmd)

    os.system("python3 -m nuitka main.py --output-dir=compile --remove-output --no-pyi-file")

    os.makedirs("compile/database")
    os.makedirs("compile/dataset/photos")

    cpfiles = ["face_recognition_sdk/config/config.yaml", "compile/face_recognition_sdk/config/config.yaml",
        "face_recognition_sdk/modules/detection/retinaface/config.yaml", "devconfig.json"]

    for cpfile in cpfiles:
        newpath = os.path.join("compile", cpfile)
        os.makedirs(os.path.dirname(newpath), exist_ok=True)
        copyfile(cpfile, newpath)
    
nuitka_compile()