import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

class HousePricePredictor:
    def __init__(self):
        # Load the training data
        self.data = pd.read_csv('train.csv')

        # Define features and target
        self.features = ['MSZoning', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
        self.target = 'Sale Price'  # Corrected from 'Sale Price' to 'SalePrice'

        # Fill missing values
        for column in self.features:
            if self.data[column].dtype == 'object':
                self.data[column] = self.data[column].fillna(self.data[column].mode()[0])
            else:
                self.data[column] = self.data[column].fillna(self.data[column].median())

        # Split data into features and target
        self.numeric_features = self.data[self.features].select_dtypes(include=['int64', 'float64']).columns
        self.categorical_features = self.data[self.features].select_dtypes(include=['object']).columns

        X = self.data[self.features]
        y = self.data[self.target]

        # Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Define preprocessor
        numeric_transformer = 'passthrough'
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

        # Define model pipeline
        self.model = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', LinearRegression())])

    def train(self):
        # Train the model
        self.model.fit(self.X_train, self.y_train)

        # Save the model to a file in /app directory
        joblib.dump(self.model, '/app/house_price_predictor.pkl')

    def predict(self, features):
        # Load the model from the /app directory
        model = joblib.load('/app/house_price_predictor.pkl')

        # Predict the price
        return model.predict(features)

# Create the predictor object and train the model
predictor = HousePricePredictor()
predictor.train()
