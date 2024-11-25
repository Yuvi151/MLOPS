import streamlit as st
import requests

# Title of the App
st.title("Weather Information App")

# Input: City name
city_name = st.text_input("Enter the city name:")

if city_name:
    # Fetch weather data from WeatherAPI
    api_key = "1cf22736d31341b09ed140310242511"  # Replace with your WeatherAPI key
    base_url = "http://api.weatherapi.com/v1/current.json"
    params = {
        "q": city_name,
        "key": api_key
    }

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        # Parse JSON response
        data = response.json()
        st.write("### Weather Details")
        st.write(f"**City:** {data['location']['name']}")
        st.write(f"**Temperature:** {data['current']['temp_c']} °C")
        st.write(f"**Weather:** {data['current']['condition']['text']}")
        st.write(f"**Humidity:** {data['current']['humidity']}%")
        st.write(f"**Wind Speed:** {data['current']['wind_kph']} kph")
    else:
        st.error("City not found. Please enter a valid city name.")

# Footer
st.write("---")
st.write("Weather data provided by [WeatherAPI](https://www.weatherapi.com/).")
