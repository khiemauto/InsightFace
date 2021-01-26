## Compile
Compile code
```
$ python3 compile.py
```
## Run
Run program

Option 1 (run without docker):
```
$ pip3 install -r requirements.txt
$ python3 main.py -rdb 1 -dbp database -fp dataset/photos
```
Option 2 (run in docker):
```
$ docker pull khiemauto92/facereg:v1.0
$ sh run.sh
```