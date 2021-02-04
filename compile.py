import glob
import os
from shutil import copyfile, make_archive

def nuitka_compile():
    pyfiles = glob.glob("sdk/**/*.py", recursive=True)
    pyfiles.extend(glob.glob("api/**/*.py", recursive=True))
    pyfiles.extend(glob.glob("core/**/*.py", recursive=True))

    for name in pyfiles:
        savepath = os.path.join("compile", os.path.dirname(name))
        cmd = f"python3 -m nuitka --module {name} --output-dir={savepath} --remove-output --no-pyi-file"
        os.system(cmd)

    # os.system("python3 -m nuitka main.py --output-dir=compile --remove-output --no-pyi-file --nofollow-import-to=_virtualenv")

    os.makedirs("compile/database")
    os.makedirs("compile/dataset/photos")

    cpfiles = ["main.py", "sdk/config/config.yaml", 
        "sdk/modules/detection/retinaface/config.yaml",
        "sdk/modules/tracking/configs/deep_sort.json", 
        "devconfig.json", "run.sh"]

    for cpfile in cpfiles:
        newpath = os.path.join("compile", cpfile)
        os.makedirs(os.path.dirname(newpath), exist_ok=True)
        copyfile(cpfile, newpath)
    
    make_archive("InsightFace", 'zip', "compile")
    
nuitka_compile()