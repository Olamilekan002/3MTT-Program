# Importing necessary libraries
import pandas as pd
import joblib


def load_data(area, bedrooms, bathrooms, location, age, garage):
    """
    Creates a DataFrame from the provided inputs, converting them into the appropriate data types.

    Parameters:
    - area: Square footage of the property
    - bedrooms: Number of bedrooms
    - bathrooms: Number of bathrooms
    - location: Geographic location of the property
    - age: Age of the property in years
    - garage: Number of garage spaces

    Returns:
    - A pandas DataFrame with the input data structured for further processing
    """
    data = pd.DataFrame(
        [
            {
                "area": int(area),
                "bedrooms": int(bedrooms),
                "bathrooms": int(bathrooms),
                "location": location,
                "age": int(age),
                "garage": int(garage),
            }
        ]
    )

    return data


def fill_missing(df):
    """
    Fills missing values in the 'location' and 'age' columns of the DataFrame.

    Parameters:
    - df: The input DataFrame with potential missing values

    Returns:
    - The DataFrame with missing 'location' filled with 'Location2' and missing 'age' filled with 60
    """

    most_location = "Location2"
    age_mean = 60

    df.location = df.location.fillna(most_location)
    df.age = df.age.fillna(age_mean)

    return df


def feature_engineering(df):
    """
    Enhances the DataFrame with new features derived from existing data for improved model predictions.

    Parameters:
    - df: The input DataFrame

    Returns:
    - A DataFrame with added features including total rooms, average room area, area-age ratio, and age-bedroom ratio.
      It also includes one-hot encoding for 'location' while dropping the original 'location' column.
    """
    # Adding the rooms feature by summing bedrooms and bathrooms
    df["rooms"] = df["bedrooms"] + df["bathrooms"]

    # Calculating average room area
    df["average_room_area"] = df["area"] / df["bedrooms"]

    # Calculating the ratio of area to age and age to number of bedrooms
    df["area_age"] = df["area"] / df["age"]
    df["age_bed"] = df["age"] / df["bedrooms"]

    # Initializing a DataFrame for one-hot encoding of locations
    location = ["Location3", "Location2", "Location5", "Location4", "Location1"]
    zero_df = pd.DataFrame([0] * 5, index=location).T

    # Setting the value to 1 for the existing location of the property
    selected_location = df["location"][0]
    zero_df[selected_location] = 1

    # Concatenating the original DataFrame with the one-hot encoded locations
    new_df = pd.concat([df, zero_df], axis=1)

    # Dropping the original 'location' column as it's now redundant
    to_drop = ["location"]
    new_df = new_df.drop(to_drop, axis=1)

    return new_df


def preprocessing(df):
    """
    Performs data preprocessing by filling missing values and feature engineering.

    Parameters:
    - df: The input DataFrame

    Returns:
    - A preprocessed DataFrame ready for model prediction
    """
    df_fill = fill_missing(df)
    df_feat = feature_engineering(df_fill)

    return df_feat


def load_model():
    """
    Loads the machine learning model from a file.

    Returns:
    - The loaded model
    """
    model = joblib.load("model.joblib")


    return model


def predict(area, bedrooms, bathrooms, location, age, garage):
    """
    Predicts the price of a property based on its features using a preloaded machine learning model.

    Parameters:
    - area, bedrooms, bathrooms, location, age, garage: Property features

    Returns:
    - The predicted price of the property
    """
    model = load_model()
    df = load_data(area, bedrooms, bathrooms, location, age, garage)
    X = preprocessing(df)

    # Predicting the price and rounding the result
    price = round(model.predict(X)[0])

    return price
