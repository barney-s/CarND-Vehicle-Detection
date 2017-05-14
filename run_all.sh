models="ycrcb2088a_32_32 ycrcb872a_32_32 ycrcb982a_32_32 ycrcb882a_32_32 ycrcb984a_32_32 ycrcb988a_32_32 rgb872a_32_32 rgb882a_32_32"


function train_all() {
  rm logs/training.log
  for model in $models; do
    echo "Training $model"
    time ./vehicle_detection.py train --params params/$model.yml --images training_data/ | tee -a logs/training.log
  done
}

function test_all() {
  rm logs/test.log
  for model in $models; do
    echo "Testing $model"
    time ./vehicle_detection.py test --model models/$model.p --config params/detection_wins_1.yml | tee -a logs/test.log
  done
}

function process_all() {
  rm logs/process.log
  for model in $models; do
    echo "Processing $model"
    time ./vehicle_detection.py process --model models/$model.p --config params/detection_wins_1.yml test_video.mp4 | tee -a logs/process.log
  done
}

if [ "$1" == "train" ]; then
   train_all
elif [ "$1" == "test" ]; then
   test_all
elif [ "$1" == "process" ]; then
   process_all
else
   echo "usage: ./run_all.sh train|test|process"
fi
