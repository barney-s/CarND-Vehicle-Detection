models="ycrcb2088a_32_32 ycrcb872a_32_32 ycrcb982a_32_32 ycrcb882a_32_32 ycrcb984a_32_32 ycrcb988a_32_32 rgb872a_32_32 rgb882a_32_32"


function train_all() {
  rm training.log
  for model in $models; do
    echo "Training $model"
    time ./vehicle_detection.py train --model $model --images training_data/ | tee -a training.log
  done
}

function test_all() {
  rm test.log
  for model in $models; do
    echo "Testing $model"
    time ./vehicle_detection.py test --model $model --config detection_wins_1 | tee -a test.log
  done
}

function process_all() {
  rm process.log
  for model in $models; do
    echo "Processing $model"
    time ./vehicle_detection.py process --model $model test_video.mp4 | tee -a process.log
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
