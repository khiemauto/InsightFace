import glob
import os

def nuitka_compile():
    pyfiles = glob.glob("face_recognition_sdk/**/*.py", recursive=True)
    pyfiles.extend(glob.glob("api/**/*.py", recursive=True))
    pyfiles.extend(glob.glob("core/**/*.py", recursive=True))
    pyfiles.append("insight_face.py")
    pyfiles.append("share_param.py")
    for name in pyfiles:
        savepath = os.path.join("compile", os.path.dirname(name))
        cmd = f"python3 -m nuitka --module {name} --output-dir={savepath}"
        os.system(cmd)
    os.system("python3 -m nuitka main.py --output-dir=compile")
    
nuitka_compile()