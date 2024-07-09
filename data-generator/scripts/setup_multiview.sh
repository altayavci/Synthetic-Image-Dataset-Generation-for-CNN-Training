cd /data-generator

python3.10 -m venv multiview
. multiview/bin/activate
pip install -r requirements.multiview.txt
deactivate

python3.10 -m venv general
. general/bin/activate
pip install -r requirements.txt
deactivate