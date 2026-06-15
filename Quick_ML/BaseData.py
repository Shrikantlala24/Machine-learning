# alright so basically, what I want is to build the functions
# that can help me build the streamlit app

# these functions are :-

# 1. getting numerical and categorical cols
# 2. creating modular code for all the techniques, also grouping them in order to keep the code clean
#       - classes for different class of techniques, but creating them as static => 
#       - 
# 3. 
# from pydantic import BaseModel


from typing import List, Annotated, Literal, Tuple
import io
import pandas as pd
# from ydata_profiling import ProfileReport

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

from pydantic import BaseModel, Field

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
    
    # def pandas_profile(self):
    #     profile=  ProfileReport(self.df, title="your profile Report", explorative=True)
    #     profile.to_file("profile.html")
    #     return "the report has been saved to 'profile.html'"

class pipe_config(BaseModel):
    test_size: Annotated[float, Field(gt=0, lt=1)] = 0.2
    random_state: int = 42

class feature_engineering_config(BaseModel):
    scaler: Literal['standard', 'minmax', 'robust'] = 'standard'
    encoder: Literal['ordinal','onehot', 'label'] = 'ordinal'
    missing_values_strategy: Literal['drop_column','drop_samples', 'mean', 'median', 'mode'] = 'mean'
    outlier_detection_strategy: Literal['z_score', 'iqr', 'none'] = 'iqr'
    outlier_handling_strategy: Literal['remove', 'cap', 'none'] = 'remove'

class feature_engineering:
    def __init__(self, feature : pd.Series):
        self.feature = feature

    # here we'll also get if the feature is numerical or categorical, and then based on that we'll apply the techniques

    def Scaling(feature_engineering_config: feature_engineering_config):
        if feature_engineering_config.scaler == 'standard':
            return StandardScaler()
        elif feature_engineering_config.scaler == 'minmax':
            return MinMaxScaler()
        elif feature_engineering_config.scaler == 'robust':
            return RobustScaler()
    
    def Encoding():
        pass
    
    def Missing_value():
        pass
    
    def Outlier_handling():
        pass

class pipeline:
