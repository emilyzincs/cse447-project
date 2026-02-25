To test

## N Gram

```
python src/myprogram.py train --work_dir work
python src/myprogram.py test --work_dir work --test_data example/input.txt --test_output mypred.txt
python grader/grade.py mypred.txt example/answer.txt
```

```
python src/myprogram.py train --work_dir work
python src/myprogram.py test --work_dir work --test_data example/input2.txt --test_output mypred.txt
python grader/grade.py mypred.txt example/answer2.txt
```

## Transformer

```
python src/transformer_model.py train --work_dir work --batch_size 128 --epochs 5 --d_model 256 --nhead 4 --num_layers 2 --max_len 128
python src/transformer_model.py test --work_dir work --test_data example/input.txt --test_output transformer_pred.txt --max_len 256
python grader/grade.py transformer_pred.txt example/answer.txt
```