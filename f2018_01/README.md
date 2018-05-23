
# first model weights:
    -t -w /home/lameeus/data/general/weights/simple/weights-03.h5
  
# 'art' model
    -w /home/lameeus/data/general/weights/simple2/weights-best.h5 --epochs 100

# Running tensorboard
in cmd

    cd ./logs
    tensorboard --logdir=./
