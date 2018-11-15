# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 13:48:23 2017

@author: Muddassar Sharif
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
import pickle
from datetime import datetime
import math
from sklearn.model_selection import train_test_split
import datetime
import io

class preprocess():

    def __init__(self):
        
        #parameters
        self.p={ 'address':None, 'variables':np.array([]), 'selected_variables':np.array([]), 'y_variable': None, 'key':{}, 'io_dim':[[],[]], 'convert':False}
        
        self.data= 0  
        self.data_y=0


          
        
    def set_parameter(self, key, value):
        self.p[key]= value

    def get_parameter(self, key=None):
        if key==None:
            return self.p
        return self.p[key]
        
              
    def read_file(self, nrows=None, start=0):   # checked
        try:
            if start== 0:
                
                if nrows== None:
                    try:
                        self.data= pd.read_csv(self.p['address'])
                    except:
                        self.data = pd.read_excel(self.p['address'])

                    return len(self.data.index)
                else:
                    self.data= pd.read_csv(self.p['address'], nrows= nrows)
                    print("Read file done!")
                    return len(self.data.index)
            else:
                self.data= pd.read_csv(self.p['address'])
                index= len(self.data.index)
                self.data = self.data[start:]
                print("Read file done!")
                return index
                
        except:
            print("File does not exist. Please check the file address.")
            return 0
    
    def variables(self):
        self.p['variables']= np.array(self.data.columns.values.tolist())
        self.p['selected_variables']= self.p['variables']
        print("Variables successfully selected")
        return self.p['selected_variables']
            
    def extract_variables(self, variables=None):
        if variables != None:
            self.p['selected_variables']= np.array(variables)
        
        if self.p['convert']== True:
            self.data=self.data[np.append(self.p['selected_variables'],  self.p['y_variable'])]
        return self.p['selected_variables']
    
  
    def replace_nan(self, nan):
        try:
            
            self.data= self.data.fillna(nan)
            print("Nan replaced with 0. Can pass something other than 0")
            return " replace nan done!"
        except:
            print("sorry an error occured!" )
            return " Sorry replace nan not done"
        
    
    def split_time(self):
        self.time= None
        temp=[]
        for col in self.p['selected_variables']: 
            if self.data[col].dtype == 'object':
                try:
                    temp2 = pd.to_datetime(self.data[col][:10])
                    temp.append(col)
                except ValueError:
                    pass
                
        
        if temp== []:
            print("File does not need time splitting")
            return "files does not have the time attribute"
        else:
            self.time= temp[0]
            temp= pd.to_datetime(self.data[self.time])
            
             
     
            self.data[self.time+ 'year']= temp.dt.year
            self.data[self.time+ 'month']= temp.dt.month
            self.data[self.time + 'day']= temp.dt.day
            if self.p['convert']== False:  # limit to running only the first time
            
                self.p['selected_variables']= np.append(self.p['selected_variables'], self.time+ 'year')
                self.p['selected_variables']= np.append(self.p['selected_variables'], self.time+ 'month')
                self.p['selected_variables']= np.append(self.p['selected_variables'], self.time+ 'day')
                self.p['variables']= np.append(self.p['variables'], self.time+ 'year')
                self.p['variables']= np.append(self.p['variables'], self.time+ 'month')
                self.p['variables']= np.append(self.p['variables'], self.time+ 'day')
                
                i= np.where(self.p['selected_variables']==self.time)
                self.p['selected_variables']= np.delete(self.p['selected_variables'], i[0][0], None)
                
    
            self.data=self.data.drop(self.time, axis=1)
            
            print("Removed and splitted " +self.time +"!")
            return "removed and splitted " +self.time +"!"

        
    def choose_y(self):
        y= self.p['y_variable']
        temp= self.y_to_float(y)
        self.p['y_variable']= y
        self.data_y= np.array(self.data[y])
        
        self.data=self.data.drop(y, axis=1)
        
        if self.p['convert']== False:
            i= np.where(self.p['selected_variables']==y)
    
            self.p['selected_variables']= np.delete(self.p['selected_variables'], i[0][0], None)
        print(str(y)+" as y variable successfully chosen")
        return "y choosen" 
    
    def y_to_float(self, y):
        try:
            self.data[y]= self.data[y].str.replace(",", "")
            self.data[y]= self.data[y].str.replace("(", "")
            self.data[y]= self.data[y].str.replace(")", "").astype(float)
            return " y converted to float "
        
        except:
            return " Error while converting y to float"
        
        
    def category_to_nominal(self):
        if self.p['convert']==False:
            self.convert= self.make_key()
        
        for x in self.p['selected_variables']:

            if (type(np.array(self.data[x])[1])== np.int64 ) or (type(np.array(self.data[x])[1])== np.float64 ) or (type(np.array(self.data[x])[1])== int ):
                pass
            else:
                self.data[[x]]=self.nom_convert_int(self.data[[x]],x)
        print("Categoricol to nominal done!")
        return "categoricol to nominal done!"
            
    def nom_convert_int(self,df,x):     
        import numpy as np
        counter=0
        x_c= np.array(df) # current column that is undergoing conversion
        dic=self.p['key'][x]
        for i in range(len(x_c)):
            
            try:
                x_c[i][0]= int(x_c[i][0])
            except:
                pass
            
            if x_c[i][0] in dic:
                pass
            else:
                dic[x_c[i][0]]=counter
                counter=counter+1
        for j in range(len(x_c)):
             x_c[j][0]=dic[x_c[j][0]]
        
        self.p['key'][x]=dic
        return x_c
        
    def make_key(self):
        for i in self.p['selected_variables']:
            self.p['key'][i]={}
            
        self.p['convert']= True  
        return True
    
    def calculate_dim(self):
        for x in self.p['selected_variables']:
            temp=(int(max(len(self.data[x].unique()), self.data[x].max()))+5)
            self.p['io_dim'][0].append(temp)
            if temp>= 8:   
                self.p['io_dim'][1].append(int(temp*.025+3))
            else:
                self.p['io_dim'][1].append(temp-1)
        print("Dim calculated")        
        return "dimensions calculated"


    def filterin(self, row, value):
        try:
            self.data = self.data.loc[self.data[row]==value]
            print("data only contain rows with: " + str(row) +" == " + str(value))
            return "data only contain rows with: " + str(row) +" == " + str(value)
        except:
            return "please check your inputs pleases. Filteration not done"
        
        
    def delete_attribute(self,column):
        try:
            self.data=self.data.drop(column, axis=1) 
            i= np.where(self.p['selected_variables']==column)
            self.p['selected_variables'] = np.delete(self.p['selected_variables'], i[0][0], None)
            print("column " +str(column) +" deleted")
            return "column " +str(column) +" deleted"
        except:
            print("column " +str(column) +" does not exist")
            return "column " +str(column) + " does not exist"

    def Filterout(self, row, value):
        try:
            self.data = self.data.loc[self.data[row]!=value]
            return "data only contain rows with: " + str(row) +" != " + str(value)
        except:
            return "please check your inputs please. Filteration not done"
        
    
    def row_aggregation(self):
        try:
            self.data= self.data.groupby(['Year', 'Month', 'Member Description', 'Description','Product Range','Sub-Category','Category','Need State'  ], as_index=False).sum()  
            return "row aggregation done"
        except:
            return "error while performing row aggregation"

    
    def Filterout_product2(self, row, value,row2,value2, y):
        try:
            self.productdata = self.data.loc[self.data[row]==value]
            self.pre_x= self.productdata.loc[self.data[row2]==value2]
            self.data.drop(self.pre_x.index, axis=0,inplace=True)
            self.productdata.drop(self.pre_x.index, axis=0,inplace=True)
            
            for i in range(50):
                self.data= self.data.append(self.productdata, ignore_index=True)
            
            self.pre_y= np.array(self.pre_x[y])
        
            self.pre_x=self.pre_x.drop(y, axis=1)
            return "data for prediction containing rows with: " + str(row) +" != " + str(value)
        except:
            return "please check your inputs please. Filteration of new product data not done"
        

        
        
    def Binning(self, inp):# not tested yet
        try:
            
            if inp==None:
                for x in self.list_s_variables:
                    if (type(self.data[x][1])== np.int64 ) or (type(self.data[x][1])== np.float64 ):
        #                    #binning when required
                            if (self.data[[x]].max()-self.data[[x]].min()) > 40:
                                length= 20
                                group_names = [i for i in range(1,length+1)]
                                jump= (self.data[[x]].max()-self.data[[x]].min())/length+ (self.data[[x]].max()-self.data[[x]].min())%length
                                number=self.data[[x]].min()                
                                bins=[]
                                for j in range(len(length)+1):              
                                    bins.append(number)
                                    number= number+jump
                                
                                bins[0]= bins[0]-10
                                bins[-1]= bins[-1] + 10
                                self.data[x] = pd.cut(self.data[x], bins, labels=group_names)
                    else:
                        pass
            
            else:
                if (self.data[[inp]].max()-self.data[[inp]].min()) > 40:
                    length= 20
                    group_names = [i for i in range(1,length+1)]
                    jump= (self.data[[x]].max()-self.data[[x]].min())/length+ (self.data[[x]].max()-self.data[[x]].min())%length
                    number=self.data[[x]].min()                
                    bins=[]
                    for j in range(len(length)+1):              
                        bins.append(number)
                        number= number+jump
                                
                        bins[0]= bins[0]-10
                        bins[-1]= bins[-1] + 10
                        self.data[x] = pd.cut(self.data[x], bins, labels=group_names)
            return "Binning Done!"
        except:
            return "An error occured when Binning"
        
        
        
    def test_train_divide(self, l):
        self.train_x , self.test_x, self.train_y ,self.test_y= train_test_split(np.array(self.data),np.array(self.y), test_size=l)
    
    
    def save_data(self, path): # no need to store the data man!
        self.data.to_pickle(path, compression='infer')
        return path
    