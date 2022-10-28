#!/bin/bash
repo_name=$(basename "$PWD")
cd ..
if [ ! -d ./qcware_env ]
then
python3.8 -m venv qcware_env
fi
source qcware_env/bin/activate
pip install ipykernel
python3.8 -m ipykernel install --user --name=qcware_env
cat << __PY38_KERNEL_JSON__ > /home/omniai-jupyter/.local/share/jupyter/kernels/qcware_env/kernel.json
{
"argv": [
"/opt/omniai/work/instance1/jupyter/qcware_env/bin/python3.8",
"-m",
"ipykernel_launcher",
"-f",
"{connection_file}"
],
"display_name": "qcware_env",
"language": "python",
"env" : {"PYTHONPATH" : "/opt/omniai/work/instance1/jupyter:/opt/omniai/work/instance1/jupyter/omniq:/opt/omniai/work/instance1/jupyter/omniq/algorithms"}
}
__PY38_KERNEL_JSON__
if [ ! -f /home/omniai-jupyter/bashrcupdated ]
then
touch /home/omniai-jupyter/bashrcupdated
cat << __BASHRC__ >> /home/omniai-jupyter/.bashrc
source /opt/omniai/work/instance1/jupyter/qcware_env/bin/activate
__BASHRC__
fi
pip install --upgrade pip
pip install -r "$repo_name"/requirements.txt
