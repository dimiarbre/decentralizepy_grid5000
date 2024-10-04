set -x
remote=$1
if [[ "$remote" == "jean-zay" ]]; then
    rsync -avzP --info=progress2 \
    --exclude '\\.*' --exclude datasets\
    --exclude attacks/my_results\
    --exclude attacks/assets\
    --exclude attacks/datasets\
    --exclude 'venv*'\
    --exclude '**/.mypy_cache'\
    --exclude '**/.git'\
    --exclude '\*/tmp*' --exclude '**__pycache__'\
    ../decentralizepy_grid5000 $remote:work/
else
    rsync -avzP --info=progress2 \
    --include '\*/decentralizepy/datasets/'\
    --exclude '\\.*' \
    --exclude 'datasets/' \
    --exclude 'attacks/my_results' \
    --exclude 'attacks/assets/' \
    --exclude 'attacks/datasets' \
    --exclude '*.sif' \
    --exclude 'venv*'\
    --exclude '\*/tmp*' \
    --exclude '**__pycache__'\
    ../decentralizepy_grid5000 $remote.g5k:scratch/
fi