from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split



def load_data_from_database():
    '''
    connects remote datasource to pandas
    load madelon data
    queries and sorts data
    return madelon_df
    '''
    url = 'postgresql://dsi:correct horse battery staple@joshuacook.me:5432'
    engine = create_engine(url)
    df = pd.read_sql("SELECT * FROM madelon", con=engine)
    return df

def add_to_process_list(process, data_dict):
    if 'processes' in data_dict.keys():
        data_dict['processes'].append(process)
    else:
        data_dict['processes'] = [process]
    
    return data_dict
   
def make_data_dict(df, random_state=None): 
    '''
    receives a dataframe
    perform a test train split
    return X_test, X_train, y_test, y_train data dictionary
    '''
    y = df['label']
    X = df.drop('label', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    data_dict = {
                 'X_test' : X_test,
                 'X_train' : X_train,
                 'y_test' : y_test,
                 'y_train' : y_train} 
    
    return data_dict

def general_transformer(transformer, data_dict):
    '''
    receives a data dictionary
    fits on train data
    transform the train and test data 
    return a data dictionary with updated train and test data and transformer
    '''
    transformer.fit(data_dict['X_train'], data_dict['y_train'])
    data_dict['X_train'] = transformer.transform(data_dict['X_train'])
    data_dict['X_test'] = transformer.transform(data_dict['X_test'])
    data_dict['transform'] = transformer
    add_to_process_list(transformer, data_dict)
    
    return data_dict
    
    
def general_model(model, data_dict):
    '''
    receives a data dictionary
    fits on training data
    scores on train and test data
    return data dictionary with model, test score, train score
    '''
    model.fit(data_dict['X_train'], data_dict['y_train'])
    train_score = model.score(data_dict['X_train'], data_dict['y_train'])
    test_score = model.score(data_dict['X_test'], data_dict['y_test'])
    add_to_process_list(model, data_dict)
    
    return {'model': model,
            'train_score': train_score,
            'test_score': test_score}
    




