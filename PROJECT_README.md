To test

## N Gram

```
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
python src/myprogram.py train --work_dir work
python src/myprogram.py test --work_dir work --test_data example/input2.txt --test_output mypred.txt
python grader/grade.py mypred.txt example/answer2.txt
```

## Transformer

```
python src/transformer_model.py train --work_dir work --batch_size 128 --epochs 15 --d_model 192 --nhead 6 --num_layers 4 --max_len 64
python src/transformer_model.py test --work_dir work --test_data example/input.txt --test_output transformer_pred.txt --max_len 64
python grader/grade.py transformer_pred.txt example/answer.txt

mkdir -p output
docker build -t cse447-proj/demo -f Dockerfile .
docker run --rm -v $PWD/src:/job/src -v $PWD/work:/job/work -v $PWD/example:/job/data -v $PWD/output:/job/output cse447-proj/demo bash /job/src/predict.sh /job/data/input2.txt /job/output/pred.txt
bash grader/grade2.sh ./example
```