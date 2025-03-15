import os
import pickle
import warnings
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st
import sklearn
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,   
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    PowerTransformer,
    FunctionTransformer
)

from feature_engine.outliers import Winsorizer
from feature_engine.datetime import DatetimeFeatures
from feature_engine.selection import SelectBySingleFeaturePerformance
from feature_engine.encoding import (
    RareLabelEncoder,
    MeanEncoder,
    CountFrequencyEncoder
)

# Configure sklearn to output pandas DataFrames
sklearn.set_config(transform_output="pandas")


# Preprocessing operations
air_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("grouper", RareLabelEncoder(tol=0.1, replace_with="Other", n_categories=2)),
    ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])

# Date of journey features
feature_to_extract = ["month", "week", "day_of_week", "day_of_year"]

doj_transformer = Pipeline(steps=[
    ("dt", DatetimeFeatures(features_to_extract=feature_to_extract, yearfirst=True, format="mixed")),
    ("scaler", MinMaxScaler())
])

# Source & destination
location_pipe1 = Pipeline(steps=[
    ("grouper", RareLabelEncoder(tol=0.1, replace_with="Other", n_categories=2)),
    ("encoder", MeanEncoder()),
    ("scaler", PowerTransformer())
])

def is_north(x):
    columns = x.columns.to_list()
    north_cities = ["Delhi", "Kolkata", "Mumbai", "New Delhi"]
    return (
        x
        .assign(**{
            f"{col}_is_north": x.loc[:, col].isin(north_cities).astype(int)
            for col in columns
        })
        .drop(columns=columns)
    )

location_transformer = FeatureUnion(transformer_list=[
    ("part1", location_pipe1),
    ("part2", FunctionTransformer(func=is_north))
])

# Departure time & arrival time
time_pipe1 = Pipeline(steps=[
    ("dt", DatetimeFeatures(features_to_extract=["hour", "minute"])),
    ("scaler", MinMaxScaler())
])

def part_of_day(x, morning=4, noon=12, eve=16, night=20):
    columns = x.columns.to_list()
    x_temp = x.assign(**{
        col: pd.to_datetime(x.loc[:, col]).dt.hour
        for col in columns
    })

    return (
        x_temp
        .assign(**{
            f"{col}_part_of_day": np.select(
                [x_temp.loc[:, col].between(morning, noon, inclusive="left"),
                 x_temp.loc[:, col].between(noon, eve, inclusive="left"),
                 x_temp.loc[:, col].between(eve, night, inclusive="left")],
                ["morning", "afternoon", "evening"],
                default="night"
            )
            for col in columns
        })
        .drop(columns=columns)
    )

time_pipe2 = Pipeline(steps=[
    ("part", FunctionTransformer(func=part_of_day)),
    ("encoder", CountFrequencyEncoder()),
    ("scaler", MinMaxScaler())
])

time_transformer = FeatureUnion(transformer_list=[
    ("part1", time_pipe1),
    ("part2", time_pipe2)
])

# Duration
class RBFPercentileSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None, percentiles=[0.25, 0.5, 0.75], gamma=0.1):
        self.variables = variables
        self.percentiles = percentiles
        self.gamma = gamma

    def fit(self, x, y=None):
        if not self.variables:
            self.variables = x.select_dtypes(include="number").columns.to_list()

        self.reference_values_ = {
            col: (
                x
                .loc[:, col]
                .quantile(self.percentiles)
                .values
                .reshape(-1, 1)
            )
            for col in self.variables
        }

        return self

    def transform(self, x):
        objects = []
        for col in self.variables:
            columns = [f"{col}_rbf_{int(percentile * 100)}" for percentile in self.percentiles]
            obj = pd.DataFrame(
                data=rbf_kernel(x.loc[:, [col]], Y=self.reference_values_[col], gamma=self.gamma),
                columns=columns
            )
            objects.append(obj)
        return pd.concat(objects, axis=1)
    

def duration_category(x, short=180, med=400):
    return (
        x
        .assign(duration_cat=np.select([x.duration.lt(short),
                                        x.duration.between(short, med, inclusive="left")],
                                       ["short", "medium"],
                                       default="long"))
        .drop(columns="duration")
    )

def is_over(x, value=1000):
    return (
        x
        .assign(**{
            f"duration_over_{value}": x.duration.ge(value).astype(int)
        })
        .drop(columns="duration")
    )

duration_pipe1 = Pipeline(steps=[
    ("rbf", RBFPercentileSimilarity()),
    ("scaler", PowerTransformer())
])

duration_pipe2 = Pipeline(steps=[
    ("cat", FunctionTransformer(func=duration_category)),
    ("encoder", OrdinalEncoder(categories=[["short", "medium", "long"]]))
])

duration_union = FeatureUnion(transformer_list=[
    ("part1", duration_pipe1),
    ("part2", duration_pipe2),
    ("part3", FunctionTransformer(func=is_over)),
    ("part4", StandardScaler())
])

duration_transformer = Pipeline(steps=[
    ("outliers", Winsorizer(capping_method="iqr", fold=1.5)),
    ("imputer", SimpleImputer(strategy="median")),
    ("union", duration_union)
])

# Total stops
def is_direct(x):
    return x.assign(is_direct_flight=x.total_stops.eq(0).astype(int))

total_stops_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("direct", FunctionTransformer(func=is_direct))
])

# Additional info
info_pipe1 = Pipeline(steps=[
    ("group", RareLabelEncoder(tol=0.1, n_categories=2, replace_with="Other")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

def have_info(x):
    return x.assign(additional_info=x.additional_info.ne("No Info").astype(int))

info_union = FeatureUnion(transformer_list=[
    ("part1", info_pipe1),
    ("part2", FunctionTransformer(func=have_info))
])

info_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
    ("union", info_union)
])

# Column transformer
column_transformer = ColumnTransformer(transformers=[
    ("air", air_transformer, ["airline"]),
    ("doj", doj_transformer, ["date_of_journey"]),
    ("location", location_transformer, ["source", 'destination']),
    ("time", time_transformer, ["dep_time", "arrival_time"]),
    ("dur", duration_transformer, ["duration"]),
    ("stops", total_stops_transformer, ["total_stops"]),
    ("info", info_transformer, ["additional_info"])
], remainder="passthrough")

# Feature selector
estimator = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)

selector = SelectBySingleFeaturePerformance(
    estimator=estimator,
    scoring="r2",
    threshold=0.1
) 

# Preprocessor
preprocessor = Pipeline(steps=[
    ("ct", column_transformer),
    ("selector", selector)
])


def train_and_save_model():
    """Train the model and save preprocessor if files don't exist"""
    if not os.path.exists("preprocessor.joblib"):
        # Read the training data
        train = pd.read_csv("train.csv")
        x_train = train.drop(columns="price")
        y_train = train.price.copy()

        # Fit and save the preprocessor
        preprocessor.fit(x_train, y_train)
        joblib.dump(preprocessor, "preprocessor.joblib")
        return x_train
    else:
        # Just return the training data for UI options
        train = pd.read_csv("train.csv")
        return train.drop(columns="price")


def main():
    """Main function for Streamlit app"""
    st.set_page_config(
        page_title="Flight Price Prediction",
        page_icon="✈️",
        layout="wide"
    )

    st.title("Flight Price Prediction - AWS SageMaker")
    
    # Initialize and get training data for UI options
    x_train = train_and_save_model()
    
    # Create two columns for the form
    col1, col2 = st.columns(2)
    
    with col1:
        airline = st.selectbox(
            "Airline:",
            options=x_train.airline.unique()
        )
        
        doj = st.date_input("Date of Journey:")
        
        source = st.selectbox(
            "Source",
            options=x_train.source.unique()
        )
        
        destination = st.selectbox(
            "Destination",
            options=x_train.destination.unique()
        )
        
        additional_info = st.selectbox(
            "Additional Info:",
            options=x_train.additional_info.unique()
        )
    
    with col2:
        dep_time = st.time_input("Departure Time:")
        
        arrival_time = st.time_input("Arrival Time:")
        
        duration = st.number_input(
            "Duration (mins):",
            step=1,
            min_value=0
        )
        
        total_stops = st.number_input(
            "Total Stops:",
            step=1,
            min_value=0
        )
    
    # Create input dataframe
    X_new = pd.DataFrame(dict(
        airline=[airline],
        date_of_journey=[doj],
        source=[source],
        destination=[destination],
        dep_time=[dep_time],
        arrival_time=[arrival_time],
        duration=[duration],
        total_stops=[total_stops],
        additional_info=[additional_info]
    )).astype({
        col: "str"
        for col in ["date_of_journey", "dep_time", "arrival_time"]
    })
    
    # Prediction button
    if st.button("Predict"):
        try:
            # Load preprocessor
            saved_preprocessor = joblib.load("preprocessor.joblib")
            X_new_pre = saved_preprocessor.transform(X_new)
            
            # Load model and predict
            with open("xgboost-model", "rb") as f:
                model = pickle.load(f)
            X_new_xgb = xgb.DMatrix(X_new_pre)
            pred = model.predict(X_new_xgb)[0]
            
            # Display prediction
            st.success(f"The predicted price is {pred:,.0f} INR")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")


if __name__ == "__main__":
    main()