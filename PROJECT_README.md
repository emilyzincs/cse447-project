To test

```
python src/myprogram.py train --work_dir work
python src/myprogram.py test --work_dir work --test_data example/input.txt --test_output mypred.txt
python grader/grade.py mypred.txt example/answer.txt
```
