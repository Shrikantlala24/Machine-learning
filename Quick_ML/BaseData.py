# alright so basically, what I want is to build the functions
# that can help me build the streamlit app

# these functions are :-

# 1. getting numerical and categorical cols
# 2. creating modular code for all the techniques, also grouping them in order to keep the code clean
#       - classes for different class of techniques, but creating them as static => 
#       - 
# 3. 


from typing import List, Annotated, Literal, Tuple
# from pydantic import BaseModel, Field
import io
import pandas as pd
# from ydata_profiling import ProfileReport

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split

from pydantic import BaseModel, Field



# first let's define the models for better feature orchestration

class _ScalerConfig(BaseModel):
    type : Literal['standard', 'minmax', 'robust']
    # only for robust scaling, we'll consider IQR
    iqr = tuple[float, float] = (0.25, 0.75)

class _EncoderConfig(BaseModel):
    type : Literal['onehot', 'ordinal', 'label']
    # only for ordinal encoding, we'll consider the order of the categories
    order : List[str] = Field(default_factory=list)
    # for one hot encoding, we'll consider the drop parameter
    drop : Literal['first', 'if_binary', None] = None
    # for handling the unknown
    handle : Literal['error', 'ignore'] = 'error'

class _ImputerConfig(BaseModel):
    type : Literal['mean', 'median', 'most_frequent', 'constant', 'knn', 'iterative']
    # only for constant imputation, we'll consider the fill value
    fill_value : float | str = 0.0
    # only for knn imputation, we'll consider the number of neighbors
    n_neighbors : int = 5
    # only for iterative imputation, we'll consider the estimator
    estimator : Literal['bayesian_ridge', 'decision_tree', 'extra_trees', 'knn', 'linear_regression', 'ridge', 'svr'] = 'bayesian_ridge'


class _OutlierConfig(BaseModel):
    type : Literal['iqr', 'zscore']
    treatment : Literal['cap', 'remove', 'none'] = 'none'
    # only for iqr, we'll consider the range
    iqr_range : Tuple[float, float] = (0.25, 0.75)
    # only for zscore, we'll consider the threshold
    threshold : float = 3.0

class _PipeConfig(BaseModel):
    test_size : float = 0.2
    random_state : int = 42
    


class data:
    def __init__(self,df : pd.DataFrame):
        self.df = df

    def num_cols(self) -> List[str]:
        return self.df.select_dtypes(include=['number']).columns.tolist()

    def cat_cols(self) -> List[str]:
        return self.df.select_dtypes(include=['object', 'category']).columns.tolist() 
    

    def overview(self) -> Tuple[pd.DataFrame, str, pd.DataFrame]:
        
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        info_str = buffer.getvalue()
        
        return (
            self.df.head(),
            info_str,
            self.df.describe()
        )
    

