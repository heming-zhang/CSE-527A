echo 'transname'
python3 main.py sigmoid 50 50 0.01 20 20 20
python3 bostontest.py

python3 main.py tanh 50 50 0.01 20 20 20
python3 bostontest.py

python3 main.py ReLU 50 50 0.01 20 20 20
python3 bostontest.py

python3 main.py ReLU2 50 50 0.01 20 20 20
python3 bostontest.py


echo 'iter'
python3 main.py sigmoid 50 40 0.01 20 20 20
python3 bostontest.py

python3 main.py sigmoid 50 30 0.01 20 20 20
python3 bostontest.py

python3 main.py sigmoid 50 20 0.01 20 20 20
python3 bostontest.py

python3 main.py sigmoid 50 60 0.01 20 20 20
python3 bostontest.py

python3 main.py sigmoid 50 70 0.01 20 20 20
python3 bostontest.py


echo 'stepsize'
python3 main.py sigmoid 50 50 0.03 20 20 20
python3 bostontest.py

python3 main.py sigmoid 50 50 0.05 20 20 20
python3 bostontest.py

python3 main.py sigmoid 50 50 0.08 20 20 20
python3 bostontest.py

python3 main.py sigmoid 50 50 0.1 20 20 20
python3 bostontest.py


echo 'wst'
python3 main.py sigmoid 50 50 0.01 10 10 10
python3 bostontest.py

python3 main.py sigmoid 50 50 0.01 30 30 30
python3 bostontest.py

python3 main.py sigmoid 50 50 0.01 40 40 40
python3 bostontest.py

python3 main.py sigmoid 50 50 0.01 10 20 30
python3 bostontest.py

python3 main.py sigmoid 50 50 0.01 30 20 10
python3 bostontest.py

python3 main.py sigmoid 50 50 0.01 10 20 10
python3 bostontest.py

python3 main.py sigmoid 50 50 0.01 30 20 30
python3 bostontest.py

python3 main.py sigmoid 50 50 0.01 20 30 40
python3 bostontest.py

python3 main.py sigmoid 50 50 0.01 40 30 20
python3 bostontest.py

python3 main.py sigmoid 50 50 0.01 40 30 20
python3 bostontest.py


echo 'rounds'
python3 main.py sigmoid 50 40 0.01 20 20 20
python3 bostontest.py

python3 main.py sigmoid 100 40 0.01 20 20 20
python3 bostontest.py

python3 main.py sigmoid 150 40 0.01 20 20 20
python3 bostontest.py

python3 main.py sigmoid 300 40 0.01 20 20 20
python3 bostontest.py