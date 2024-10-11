set -x

rsync -avzp --info=progress2 --exclude "*.pt" --exclude "*.log" --exclude "*.npy" --exclude "**/machine*/*.png" $@