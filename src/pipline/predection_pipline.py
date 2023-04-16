import os
import sys
import numpy as np 
import pandas as pd
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException

from src.utils import load_object

class PredictPipline:
    def __init__(self):
        pass


    def predict(self,features):
        try:
            ## This line of path code work i both windos and linex
            preprocessor_path = os.path.join("artifcats","preprocessor.pkl")
            model_path = os.path.join("artifcats","model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)


            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            return pred


        except Exception as e:
            logging.info("Error Ocure in Prediction PIPLine")
            raise CustomException(e, sys)

    
class CustomData:
    def __init__(self,
                carat:float,
                depth:float,
                table:float,
                x:float,
                y:float,
                z:float,
                cut:str,
                color:str,
                clarity:str):

        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity


    def get_data_as_data_frame(self):
        try:
            custome_data_input_dict = {
                "carat":[self.carat],
                "depth":[self.depth],
                "table":[self.table],
                "x":[self.x],
                "y":[self.y],
                "z":[self.z],
                "cut":[self.cut],
                "color":[self.color],
                "clarity":[self.clarity]
            }

            data = pd.DataFrame(custome_data_input_dict)
            logging.info("DataFrame Gathered")
            return data

        except Exception as e:
            logging.info("Error Ocured in Prediction Pipline")
            raise CustomException(e, sys)






