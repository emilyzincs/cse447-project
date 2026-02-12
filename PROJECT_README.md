To test

# 🔹 TRAIN (first)
python src/myprogram.py train --work_dir work

# PREDICT (second)
Do docker
```
mkdir -p output
docker build -t cse447-proj/demo -f Dockerfile .
docker run --rm -v $PWD/src:/job/src -v $PWD/work:/job/work -v $PWD/example:/job/data -v $PWD/output:/job/output cse447-proj/demo bash /job/src/predict.sh /job/data/input.txt /job/output/pred.txt
bash grader/grade.sh ./example
```

```
mkdir -p output
docker build -t cse447-proj/demo -f Dockerfile .
docker run --rm -v $PWD/src:/job/src -v $PWD/work:/job/work -v $PWD/example:/job/data -v $PWD/output:/job/output cse447-proj/demo bash /job/src/predict.sh /job/data/input2.txt /job/output/pred.txt
bash grader/grade2.sh ./example
```