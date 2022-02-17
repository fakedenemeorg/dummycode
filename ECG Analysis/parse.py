import pandas as pd
import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn import preprocessing 
from tensorflow import metrics

df = pd.read_csv("mitbih_train.csv")

mitbih_train = np.loadtxt('mitbih_train.csv', delimiter=',')
set(list(mitbih_train[:,-1]))



def create_model():
    model = Sequential([
        Dense(units=8,input_shape=(187,), activation='relu'),
        Dense(units=16,activation='relu'),
        Dense(units=32,activation='relu'),
        Dense(units=4,activation='relu'),
        Dense(units=5,activation='softmax')
    ]
    )
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer= 'Adam',metrics=['accuracy'])
    
    return model



def train(model):

    mitbih_train = np.loadtxt('mitbih_train.csv', delimiter=',')
    mitbih_test = np.loadtxt('mitbih_test.csv', delimiter=',')
    
    labels = mitbih_train[:,-1]
    test_label = mitbih_test[:,-1]



    mitbih_train = mitbih_train[:,:-1]
    mitbih_test = mitbih_test[:,:-1]

    mitbih_test = mitbih_test,test_label

    model.fit(mitbih_train,labels,shuffle=True, validation_data = mitbih_test,epochs = 30, batch_size=32)




def predict(model,filepath_to_read="test_data_A.csv"):
        test_df = pd.read_csv(
            "test_data_A.csv",
            sep='|',
            usecols=[11,12,13,14,15,17,18,19,20,21,22,23,24,26,27,31,33,34,35],
            header = 0,
            names=['row', 'uid', 'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id', 'inter_typ_cd', 'slot_id', 'spread_app_id', 'tags', 'app_first_class', 'app_second_class', 'age', 'city', 'city_rank', 'device_name', 'device_size', 'career', 'gender', 'net_type', 'residence', 'his_app_size', 'his_on_shelf_time', 'app_score', 'emui_dev', 'list_time', 'device_price', 'up_life_duration', 'up_membership_grade', 'membership_life_duration', 'consume_purchase', 'communication_onlinerate', 'communication_avgonline_30d', 'indu_name', 'pt_d']
            )
        
        test_df = np.array(test_df)
        
        predicted = model.predict(test_df)
            
        f = open("results.txt","w+")
        
        for i,pred in enumerate(predicted):
            f.write(str(i) + "," + str(pred[0]) + "\n")
        f.close()




model = create_model()
train(model)


