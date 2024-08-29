# Script to deploy on a jupyter notebook, for a control node that will launch jobs & perform attacks.
module load cuda; 
python -m venv /tmp/custom_python; 
source /tmp/custom_python/bin/activate; 
mkdir /tmp/cache_python
pip install --cache-dir /tmp/cache_python enoslib --editable decentralizepy/; 
mkdir /tmp/logs