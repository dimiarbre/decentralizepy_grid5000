set -x
remote=$1
if [[ "$remote" == "jean-zay" ]]; then
    rsync -avzP --info=progress2 \
    --exclude '\\.*'\
    --exclude datasets\
    --exclude attacks/my_results\
    --exclude assets\
    --exclude attacks/assets\
    --exclude attacks/datasets\
    --exclude 'venv*'\
    --exclude '**/.mypy_cache'\
    --exclude '**/.git'\
    --exclude '**/.vscode'\
    --exclude '\*/tmp*'\
    --exclude '**__pycache__'\
    ../decentralizepy_grid5000 $remote:work/
elif [[ "$remote" == "rennes" || "$remote" == "nancy" ]]; then
    rsync -avzP --info=progress2 \
    --include '\*/decentralizepy/datasets/'\
    --exclude '\\.*' \
    --exclude 'datasets/' \
    --exclude 'attacks/my_results' \
    --exclude assets\
    --exclude 'attacks/assets/' \
    --exclude 'attacks/datasets' \
    --exclude '*.sif' \
    --exclude 'venv*'\
    --exclude '**/.mypy_cache'\
    --exclude '**/.git'\
    --exclude '**/.vscode'\
    --exclude '\*/tmp*' \
    --exclude '**__pycache__'\
    ../decentralizepy_grid5000 $remote.g5k:scratch/
else
    echo "$remote not in ['jean-zay', 'rennes', 'nancy']. Select a correct remote."
fi