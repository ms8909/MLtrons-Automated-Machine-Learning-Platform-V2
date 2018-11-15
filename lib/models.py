import numpy
numpy.random.seed(123)
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn import neighbors
from sklearn.preprocessing import Normalizer
import keras
# from keras.layers import merge
from keras.layers import Merge
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Reshape
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error
from keras.models import load_model
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition
import numpy as np
import random as rd
from keras.callbacks import TensorBoard
from time import time
import os
from keras.models import model_from_json


def split_features(X):
    X_list = []
    for i in range(len(X[0])):
        
        temp = X[..., [i]]
        X_list.append(temp)
    
    return X_list


class Model(object):

    def evaluate(self, X_val, y_val):
        assert(min(y_val)+1 > 0)
        guessed_sales = self.guess(X_val)
        relative_err = numpy.absolute((y_val - guessed_sales) / y_val)
        result = numpy.sum(relative_err) / len(y_val)
        return result
    
    
    def rmspe(self, X_val, y_val):
        assert(min(y_val)+1 > 0)
        guessed_sales = self.guess(X_val)
        relative_err = (y_val - guessed_sales) / (y_val)
        r_err_sq= numpy.square(relative_err)
        result = numpy.sum(r_err_sq) / len(y_val)
        return result**.5
    
    def rmse(self, X_val, y_val):
        assert(min(y_val) +1> 0)
        guessed_sales = self.guess(X_val)
        result= mean_squared_error(y_val, guessed_sales)**.5
        return result

    def rmsle(self, X_val, y_val):
        assert(min(y_val)+1 > 0)
        guessed_sales = numpy.log(numpy.array(self.guess(X_val))+1)
        y_val= numpy.log(numpy.array(y_val)+1)
        result= mean_squared_error( guessed_sales, y_val)**.5
        return result


    
class NN_with_EE(Model):

    def __init__(self):
        super(NN_with_EE, self).__init__()
        self.param= {'nb_ephoch':10, 'checkpointer': None,'model':['weights','architecture'], 'io_dim':[[],[]], 
                     'batch_size':200 , 'model_results':{'rmspe':None, 'rmse':None, 'rmsle':None}}
        self.model= None
        
                     
    def build_model(self, project_name, io_dim):
        directroy = '../projects/'+str(project_name)+'/models'
        if not os.path.exists(directroy):
            os.makedirs(directroy)

        self.param['model'][0]= directroy+"/NNEE_weights.h5"
        self.param['model'][1]= directroy+"/NNEE_architecture.json"
        self.param['checkpointer'] = ModelCheckpoint(filepath=self.param['model'][0], verbose=1, save_best_only=True)
        self.param['io_dim']=io_dim

        self.__build_keras_model(self.param['io_dim'][0], self.param['io_dim'][1])


    def __build_keras_model(self, input_dim, output_dim):
        models = []

        for i in range(len(input_dim)):
            model_temp = Sequential()
            if input_dim[i]==1:
                model_temp.add(Dense(1, input_dim=1))
            else:    
                model_temp.add(Embedding(input_dim[i], output_dim[i], input_length=1))
                model_temp.add(Reshape(target_shape=(output_dim[i],)))
            models.append(model_temp)
            

        self.model = Sequential()
        self.model.add(Merge(models, mode='concat'))
#         self.model.add(Dense(3000, init='uniform'))
#         self.model.add(Activation('relu'))
#         self.model.add(Dropout(0.2))
        self.model.add(Dense(1000, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(200, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='mean_squared_error', optimizer='adam')

        # serialize model to JSON
        model_json = self.model.to_json()
        with open(self.param['model'][1], "w") as json_file:
            json_file.write(model_json)

           
    def load_model(self, project_name):
        directroy = '../projects/'+str(project_name)+'/models'
        if not os.path.exists(directroy):
            print("Project name is not correct")
            return False
            
        self.param['model'][0]= directroy+"/NNEE_weights.h5"
        self.param['model'][1]= directroy+"/NNEE_architecture.json"
        self.param['checkpointer'] = ModelCheckpoint(filepath=self.param['model'][0], verbose=1, save_best_only=True)

        # load json and create model
        json_file = open(self.param['model'][1], 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(self.param['model'][0])
        print("Loaded model from disk")
        # compile model
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def preprocessing(self, X):
        X_list = split_features(X)
        return X_list
    
 
    def _val_for_fit(self, val):
        ans = numpy.log(val+1) / numpy.log(10000)
        return ans

    def _val_for_pred(self, ans):
        val = numpy.exp(ans*numpy.log(10000)) -1 
        return val
    

    def fit(self, X_train, y_train, X_val, y_val):
        print("Model is training")

        self.model.fit(self.preprocessing(X_train), self._val_for_fit(y_train),
                       epochs=self.param['nb_ephoch'], batch_size=self.param['batch_size'],
                       callbacks=[self.param['checkpointer']],
                       validation_data=(self.preprocessing(X_val), self._val_for_fit(y_val)),
                       )

        self.param['model_results']['rmspe']=self.rmspe(X_val, y_val)
        self.param['model_results']['rmse']=self.rmse(X_val, y_val)
        self.param['model_results']['rmsle']=self.rmsle(X_val, y_val)
        print("Result on validation data: ",self.param['model_results'] )
        return self.param['model_results']['rmse']
        
        
    def guess(self, features):
        features = self.preprocessing(features)
        result = self.model.predict(features).flatten()
        return self._val_for_pred(result)
    
    def get_parameter(self, key=None):
        if key== None:
            return self.param
        else:
            try:
                return self.param[key]
            except:
                print(str(key)+" does not exist")
                return False
          
    def set_parameter(self, key, value):
        try:
            self.param[key]= value
        except:
            print("Something went wrong in set_parameter")
            
    
class XGBoost(Model):

    def __init__(self):
        super(XGBoost, self).__init__()
        self.param= {'nb_ephoch':1, 'first_training': False,'model':['weights','architecture'], 'io_dim':[[],[]], 'model_results':{'rmspe':None, 'rmse':None, 'rmsle':None}, 'batch_size':100 }
        self.model= None
        
    def build_model(self, project_name, io_dim=None):
        directroy = '../projects/'+str(project_name)+'/models'
        if not os.path.exists(directroy):
            os.makedirs(directroy)

        self.param['model'][0]= directroy+"/xgb_model.model"
        self.param['model'][1]= directroy+"/xgb_architecture.json"
        self.param['io_dim']=io_dim
        
        
    def fit(self, X_train, y_train, X_val, y_val):
#         self.load_preprocess_object()
        dtrain = xgb.DMatrix(X_train, label=numpy.log(y_train+1))
        evallist = [(dtrain, 'train')]
        param = {'nthread': -1,
                 'max_depth': 7,
                 'eta': 0.02,
                 'silent': 1,
                 'objective': 'reg:linear',
                 'colsample_bytree': 0.7,
                 'subsample': 0.7}
        num_round = 10000
        if self.param['first_training']==False:
            self.model = xgb.train(param, dtrain, num_round, evallist)
            self.param['first_training']=True
            self.model.save_model(self.param['model'][0])
        else:
            self.model = xgb.train(param, dtrain, num_round, evallist,xgb_model=self.param['model'][0])
            self.model.save_model(self.param['model'][0])
            
        self.param['model_results']['rmspe']=self.rmspe(X_val, y_val)
        self.param['model_results']['rmse']=self.rmse(X_val, y_val)
        self.param['model_results']['rmsle']=self.rmsle(X_val, y_val)
        print('RMSE is ', self.param['model_results'])
        return self.param['model_results']['rmse']
            
    def guess(self, feature):
        dtest = xgb.DMatrix(feature)
        return numpy.exp(self.model.predict(dtest))-1
    
    def get_parameter(self, key=None):
        if key== None:
            return self.param
        else:
            try:
                return self.param[key]
            except:
                print(str(key)+" does not exist")
                return False
          
    def set_parameter(self, key, value):
        try:
            self.param[key]= value
        except:
            print("Something went wrong in set_parameter")
 
  
class time_series(Model):

    def __init__(self):
        super(time_series, self).__init__()
        self.param= {'nb_ephoch':100, 'checkpointer': None,'model':['weights','architecture'], 'io_dim':[[],[]], 'model_results':None, 'batch_size':1, 'window':20 }
        self.model= None
        
                     
    def build_model(self, project_name, ephoch=100):
        directroy = '../projects/'+str(project_name)+'/models'
        if not os.path.exists(directroy):
            os.makedirs(directroy)

        self.param['model'][0]= directroy+"/LSTM_weights.h5"
        self.param['model'][1]= directroy+"/LSTM_architecture.json"
        self.param['checkpointer'] = ModelCheckpoint(filepath=self.param['model'][0], verbose=1, save_best_only=True)
        self.param['nb_ephoch']= ephoch

        self.__build_keras_model()


    def __build_keras_model(self):
        self.model = Sequential()
        self.model.add(LSTM(4, input_shape=(1, self.param['window'])))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

        # serialize model to JSON
        model_json = self.model.to_json()
        with open(self.param['model'][1], "w") as json_file:
            json_file.write(model_json)

           
    def load_model(self, project_name):
        directroy = '../projects/'+str(project_name)+'/models'
        if not os.path.exists(directroy):
            print("Project name is not correct")
            return False
            
        self.param['model'][0]= directroy+"/NNEE_weights.h5"
        self.param['model'][1]= directroy+"/NNEE_architecture.json"
        self.param['checkpointer'] = ModelCheckpoint(filepath=self.param['model'][0], verbose=1, save_best_only=True)
 

        # load json and create model
        json_file = open(self.param['model'][1], 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(self.param['model'][0])
        print("Loaded model from disk")
        # compile model
        self.model.compile(loss='mean_squared_error', optimizer='adam')

 
    def _val_for_fit(self, val):
        ans = numpy.log(val+1) / numpy.log(10000)
        return ans

    def _val_for_pred(self, ans):
        val = numpy.exp(ans*numpy.log(10000)) -1 
        return val
    
    def create_dataset(self, dataset):
        look_back= self.param['window']
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back):
            b = dataset[i:(i+look_back)]
            dataX.append(b)
            dataY.append(dataset[i + look_back])
        return numpy.array(dataX), numpy.array(dataY)
    

    def fit(self, y_train, y_test):
        print("Model is training")
        x_train, y_train= self.create_dataset(self._val_for_fit(y_train))
        x_test, y_test= self.create_dataset(self._val_for_fit(y_test))
        #need to reshape x before passing to the model
        x_train= numpy.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        x_test= numpy.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
        self.param['memory']= y_train[ self.param['window']*-1:]
        
        self.model.fit(x_train, y_train, epochs=self.param['nb_ephoch'], batch_size=self.param['batch_size'],
                       callbacks=[self.param['checkpointer']], verbose=2,
                        validation_data=(x_test, y_test))
        


#         self.param['model_results']= self.rmse(x_test, y_test)
#         print("Result on validation data: ",self.param['model_results'] )
        return self.param['model_results']
        
        
    def guess(self, num_points=30):
        memory= self.param['memory']
   
        for i in range(30):
            x_train= memory[-self.param['window']:]
            x_train= numpy.array([x_train])
            x_train= numpy.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
            result= self.model.predict(x_train)[0][0]
            memory= np.append(memory,result)
                          
        return self._val_for_pred(memory[-30:])
    
    def get_parameter(self, key=None):
        if key== None:
            return self.param
        else:
            try:
                return self.param[key]
            except:
                print(str(key)+" does not exist")
                return False
          
    def set_parameter(self, key, value):
        try:
            self.param[key]= value
        except:
            print("Something went wrong in set_parameter")
            

