import pandas as pd 
import streamlit as st 
import joblib

st.title ("F1 Race Predictor")
model = joblib.load("data/f1_model.pkl")
df =  pd.read_csv("data/df_model.csv")

driver_names = df[['forename','surname']].drop_duplicates()
driver_names['full name']=driver_names['forename']+' '+ driver_names['surname']
selected_driver =st.selectbox("Select a driver",sorted(driver_names['full name'].unique()))

driver_races = df[(df['forename'] + ' ' + df['surname']) == selected_driver]
race_options = driver_races['name'] + ' ' + driver_races['year'].astype(str)
selected_race = st.selectbox("Select a race", sorted(race_options.unique()))

race_row = driver_races[(driver_races['name'] + ' ' + driver_races['year'].astype(str))==selected_race].iloc[0]

features = ['grid', 'avg_last5_finish', 'constructor_avg_last5', 'grid_vs_form']
X_input = race_row[features].values.reshape(1, -1)
prediction_prob = model.predict_proba(X_input)[0][1]

st.subheader("Prediction")
st.metric("Top-10 Finish Probability", f"{prediction_prob:.1%}")

st.write("**Actual result:**", "Top 10" if race_row['top10_finish'] == 1 else "Not Top 10")

st.write("**Stats used for this prediction:**")
st.write(f"- Grid position: {race_row['grid']}")
st.write(f"- Recent avg finish: {race_row['avg_last5_finish']:.1f}")
st.write(f"- Constructor recent avg finish: {race_row['constructor_avg_last5']:.1f}")

st.subheader(f"{selected_driver}'s Recent Form")
driver_history = driver_races.sort_values(['year', 'raceId']).tail(10)
st.line_chart(driver_history.set_index('name')['positionOrder'])



