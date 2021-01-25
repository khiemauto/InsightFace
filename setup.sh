# wget -O model/attr_mbnet2_jit_best.pt "https://drive.google.com/uc?export=download&id=1Lr8bzdehsJHBXPabfbH8aaPJnqbXR2Z3"
# wget -O model/attr_resnet18_jit_best.pt "https://drive.google.com/uc?export=download&id=1_TnARb76rwctKg2HHv3ciVpTbyprMLzp"
# wget -O model/iresnet34-5b0d0e90.pth "https://drive.google.com/uc?export=download&id=1K6pmuGr-i1-DZe5vuiG0le0SZFkv-Mom"
# wget -O model/iresnet50-7f187506.pth "https://drive.google.com/uc?export=download&id=1ShB2CdCSfo9uYmG0-a5we_qeFtOrWK2O"
# wget -O model/iresnet100-73e07ba7.pth "https://drive.google.com/uc?export=download&id=1RBau2Z6qHfoRZ3CRz8wQjYrmrDPC6p-3"
# wget -O model/mobilenet0.25_Final.pth "https://drive.google.com/uc?export=download&id=1yYBwF6f4aECRykq64ApNf5hECalKzfku"
# wget -O model/Resnet50_Final.pth "https://drive.google.com/uc?export=download&id=1-W0_SgleuY_2yW-K3PwzSWS4-Zj8GGYX"
docker pull khiemauto92/facereg:v1.0
python3 compile.py