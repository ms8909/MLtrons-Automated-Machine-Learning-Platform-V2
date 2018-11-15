import pickle
from .models import *
from .graph import *
from .preprocess  import *
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
sys.setrecursionlimit(10000)
import datetime
import json
import plotly
from plotly import plotly as py
import plotly.graph_objs as go
import math

class Dashboard(): 

    def __init__(self, project_name, problem_type): 
        
        self.pa={'file_address':None,'y_variable':None,'preprocess_param': None, 'project_name': None, 'problem_type':None, 'model_param':None, 'mode':'train', 'best_model': 'NN_with_EE'} 
        
        self.pa['problem_type']= problem_type
        self.pa['project_name']= project_name
        
        if problem_type== 'p':  
            self.model={'xgb':XGBoost()}
#             self.model={'NN_with_EE': NN_with_EE(),'xgb':XGBoost()}
        if problem_type == 'f':
            self.model= {'time_series':time_series()}

        self.preprocess_object = preprocess() #replace this with only variables
        
        # temporary variables for ploting
        self.y_given=[]
        self.y_ans=[]
        self.y_ans_given=[]
    
    def set_parameter(self, key, value):
        self.pa[key]= value

    def get_parameter(self, key=None):
        if key==None:
            return self.pa
        return self.pa[key]
    
    def get_best_model(self):
        return self.best_model
    
    def get_trained_models(self):
        return self.models
        
    def transform_data(self, filter_in=None):
        print("reading file")
        self.preprocess_object.set_parameter('address', self.pa['file_address'])
        self.preprocess_object.set_parameter('y_variable', self.pa['y_variable'])
        self.preprocess_object.read_file()
        if filter_in!= None:
            self.preprocess_object.filterin(filter_in[0], filter_in[1])
        self.preprocess_object.replace_nan(0)
        self.preprocess_object.variables()
        self.preprocess_object.split_time()
        if self.pa['mode'] == "train": 
            self.preprocess_object.choose_y()
        self.preprocess_object.category_to_nominal()
        self.preprocess_object.calculate_dim()
        self.pa['preprocess_param']= self.preprocess_object.get_parameter()
        
    def find_best_model(self, train_test_split=.9, epoch=100, batch_size= None): 
        model_rmse=math.inf 
        split= int(len(np.array(self.preprocess_object.data))*train_test_split)
        self.y_given=np.array(self.preprocess_object.data_y[:split])
        self.y_ans_given=np.array(self.preprocess_object.data_y[split:])
        
        if self.pa['problem_type']=='f':
            self.y_given=np.array(self.preprocess_object.data_y[:split])
            self.model['time_series'].build_model(self.pa['project_name'],epoch )
            self.model['time_series'].fit(np.array(self.preprocess_object.data_y[:split]),
                                          np.array(self.preprocess_object.data_y[split:]))
            self.pa['best_model']= 'time_series'
            return True
            
  
        self.y_given=np.array(self.preprocess_object.data_y[:split])
        for model_name in self.model.keys():
            if self.pa['problem_type']=='p':
                self.model[model_name].build_model(self.pa['project_name'], self.pa['preprocess_param']['io_dim'])
                #update epoch and batch_size here
                if epoch!=None:
                    self.model[model_name].set_parameter('nb_ephoch', epoch)              
                if batch_size!=None:
                    self.model[model_name].set_parameter('batch_size', batch_size)
                    
                rmse= self.model[model_name].fit(np.array(self.preprocess_object.data[:split]),
                                 np.array(self.preprocess_object.data_y[:split]),
                                 np.array(self.preprocess_object.data[split:]), np.array(self.preprocess_object.data_y[split:]))
                #update best model vraible
                if rmse< model_rmse:
                    model_rmse= rmse
                    self.pa['best_model']= model_name
                    
        print("Training Done")
     


    def predict(self, data= None): # needs to be changed
        k= self.pa['best_model']
        if data==None:
            self.y_ans= self.model[k].guess(np.array(self.preprocess_object.data[900:]))
            return self.y_ans
        else:
            self.y_ans= self.model[k].guess(np.array(data))
            return self.y_ans

    def forecast(self, number= 30): # needs to be changed
        k= self.pa['best_model']
        self.y_ans=self.model[k].guess(number)
        return self.y_ans
 

    def get_prediction_key(self):
        #self.load_preprocess_object()
        self.key= self.preprocess_object.get_key_for_perdiction()
        return [-1, self.key[0], self.key[1]]

    
    
    def make_graph_object(self):
#        self.load_preprocess_object()
#        key= self.preprocess_object.get_key_for_perdiction()
        graph1= graph()
        self.graph_objects.append(graph1) 
        self.graph_objects[-1].key_from_user(self.key)
        return [self.graph_objects.index(self.graph_objects[-1])]
        
    


    def make_graph(self,k, user_in):
        if user_in != None:
            
            self.graph_objects[k].user_input(user_in)
       # return self.graph_objects[k].get_df_for_per()
        y= self.best_model[self.best_model.keys()[0]][0].guess(np.array(self.graph_objects[k].get_df_for_per()))  
        self.graph_objects[k].set_per_y(y)
        return [self.graph_objects.index(self.graph_objects[k]), self.graph_objects[k].get_plot_data()]

   
    def get_preporcessed_file(self):
        return self.preprocess_object

    def graph(self):

        plotly.tools.set_credentials_file(username='ms8909', api_key='5s7096UcTVWmOJoPeq8w')

        x_temp = []
        for i in range(len(self.y_given)+len(self.y_ans)):
            x_temp.append(i)
            
        data = []
        trace1 = go.Scatter(
            x=x_temp,
            y=list(self.y_given),  #list(self.y_ans_given)+ 
            mode='lines',
            name='Actual Sales'
        )
        data.append(trace1)
        
        d= [0 for i in range(len(self.y_given)-1)]
  
        trace2 = go.Scatter(
            x=x_temp,
            y=d+ list(self.y_ans),
            mode='lines',
            name='predicted Sales'
        )
        data.append(trace2)        

        
        fig = dict(data=data)
        py.iplot(fig, filename='line-mode')
        