# Demo for lung cancer diagnosis.

IS_ISBI=${IS_ISBI:-0}
GPU=${GPU:-''}

while [ $# -gt 0 ]; do
   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi
  shift
done

mkdir -p data/dicom
mkdir -p data/volumes
mkdir -p data/candidates
mkdir -p data/slices
mkdir -p output/sorted_slices
mkdir -p output/sorted_slices_images

if [ $IS_ISBI -eq 1 ]
then
    CUDA_VISIBLE_DEVICES=$GPU python src/test_ISBI.py 
else
    CUDA_VISIBLE_DEVICES=$GPU python src/test.py
fi

