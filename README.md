# Predictive Maintenance

## Setup 

* Prepare for the input data: In a csv file, each row is a time point, each column is a sensor measurement. 
* Add a 'label' column. 
* Define a time window you want to notify potential failures, such as 30 hours



-------------
## For training:
Pass 3 arguments: 
* path where training data is, 
* sliding window size, 
* names of the columns for feature extration, 
* path where you want to save your trained model

data1: 
```
python3 template.py /data/preventive_maintenance/train_turbofan/ 5 s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21 /data/preventive_maintenance/model.pickle
       python  template.py ./train_turbofan/ 5 s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21 ./turbofan_model.pickle
```
data2: 
```
python3 template.py /data/preventive_maintenance/train_bearing/ 2560 Horizontal_acceleration,Vertical_acceleration /data/preventive_maintenance/model.pickle
       python  template.py ./train_bearing/ 2560 Horizontal_acceleration,Vertical_acceleration ./bearing_model.pickle
```

------------
## For testing:
pass 2 arguments: 
*  path where you saved your trained model, 
* path where testing data is


data1: 
```
python3 test.py /data/preventive_maintenance/model.pickle /data/preventive_maintenance/test_turbofan.csv
       python  test.py ./turbofan_model.pickle ./test_turbofan.csv
```


data2: 
```
python3 test.py /data/preventive_maintenance/model.pickle /data/preventive_maintenance/test_bearing.csv
       python  test.py ./bearing_model.pickle ./test_bearing.csv
```


