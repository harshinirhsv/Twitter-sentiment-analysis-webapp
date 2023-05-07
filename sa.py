from configparser import Interpolation
from tkinter import HORIZONTAL
import streamlit as st
from streamlit_option_menu import option_menu
from  PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
##img = Image.open('SIH_2018_logo.png')
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go 
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt

##st.set_page_config(page_title="Road Analytics tool",page_icon=img)

hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            body {
            background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
            background-size: cover;
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# local_css("style.css")

# st.sidebar.image(img, use_column_width=False,width=None)      #-----------> For icon on the side
with st.sidebar:
    selected = option_menu(
        menu_title="Dashboard",
        options=["Home","Analytics","Visualization","Graphs","Text based analysis"],
        icons=["house","activity","graph-up","graph-down","pencil"],
        menu_icon=None,
        
        styles={
                "container": {"padding": "0!important", "background-color": "DBF0F9"},  #background color
                "icon": {"color": "pink", "font-size": "20px"},   # icon color
                "nav-link": {
                    "font-size": "20px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "DBF0F9", # icon color after hover
                },
                "nav-link-selected": {"background-color": "#ADD8E6"},   # Colour when tab is selected  #ADD8E6
                "icon": {"color": "blue", "font-size": "30px"},   # icon color
            },
        )




if selected=="Home":
        st.title("Twitter Sentiment Analysis")
        font="serif"
        st.write("Application of Natural Language Processing")



if selected=="Analytics":
    st.header('Upload your CSV data')
    st.write("Now it's time for you to uplod your CSV files with the drag and drop functionality")
    uploaded_file = st.file_uploader("Note: Only CSV files are allowed", type=["csv"])
# Pandas Profiling Report
    if uploaded_file is not None:
        @st.cache
        def load_csv():
            csv = pd.read_csv(uploaded_file)
            return csv
        df = load_csv()
        pr = ProfileReport(df, explorative=True)
        st.header('**The following is the Data frame that you have uploaded..**')
        st.write(df)
        st.write('---')
        st.header('**Please wait your report is being generated...**')
        st_profile_report(pr)
        pr.to_file("Analysis.html")
        pr.to_file("Analysis.json")
    else:
        st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        @st.cache
        def load_data():
            a = pd.DataFrame(
                np.random.rand(100, 5),
                columns=['a', 'b', 'c', 'd', 'e']
            )
            return a
        df = load_data()
        pr = ProfileReport(df, explorative=True)
        st.header('**Example Dataset is being loaded..**')
        st.write(df)
        st.write('---')
        st.header('**Please wait your report is being generated...**')
        st_profile_report(pr)



if selected=="Visualization":
    st.header("Visualization")
    st.write("Now its time for us to visualize the dataset that we have uploaded. When speaking about visualization, We should als understand there are many types. Some of which are as follows: \n  1) Bar chart \n 2) Scatter Plot \n 3) Line plot \n 4) Histogram")
    st.write("In the side bar use the drop down menu to visualize the dataset that we have uploaded.")

    st.set_option('deprecation.showfileUploaderEncoding', False)


# Add a sidebar
    st.sidebar.subheader("Visualization Settings")

# Setup file upload
    uploaded_file = st.sidebar.file_uploader(
        label="Upload your CSV or Excel file. (200MB max)",
        type=['csv', 'xlsx'])

    global da 
    if uploaded_file is None:
        st.title("Your file is not uploaded....")
    if uploaded_file is not None:
        print(uploaded_file)
        print("hello")
        try:
            da = pd.read_csv(uploaded_file)
        except Exception as e:
            print(e)
            da = pd.read_excel(uploaded_file)

    global numeric_columns
    global non_numeric_columns
    try:
        st.write(da)
        numeric_columns = list(da.select_dtypes(['float', 'int']).columns)
        non_numeric_columns = list(da.select_dtypes(['object']).columns)
        non_numeric_columns.append(None)
        print(non_numeric_columns)
    except Exception as e:
        print(e)
        st.write("Please upload file to the application.")

# add a select widget to the side bar
    chart_select = st.sidebar.selectbox(
        label="Select the chart type",
        options=['Scatterplots', 'Lineplots', 'Histogram', 'Boxplot']
    )

    if chart_select == 'Scatterplots':
        st.sidebar.subheader("Scatterplot Settings")
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
            plot = px.scatter(data_frame=da, x=x_values, y=y_values, color=color_value)
        # display the chart
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

    if chart_select == 'Lineplots':
        st.sidebar.subheader("Line Plot Settings")
        try:
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
            plot = px.line(data_frame=da, x=x_values, y=y_values, color=color_value)
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

    if chart_select == 'Histogram':
        st.sidebar.subheader("Histogram Settings")
        try:
            x = st.sidebar.selectbox('Feature', options=numeric_columns)
            bin_size = st.sidebar.slider("Number of Bins", min_value=10,
            max_value=100, value=40)
            color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
            plot = px.histogram(x=x, data_frame=da, color=color_value)
            st.plotly_chart(plot)
        except Exception as e:
            print(e)

    if chart_select == 'Boxplot':
        st.sidebar.subheader("Boxplot Settings")
        try:
            y = st.sidebar.selectbox("Y axis", options=numeric_columns)
            x = st.sidebar.selectbox("X axis", options=non_numeric_columns)
            color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
            plot = px.box(data_frame=da, y=y, x=x, color=color_value)
            st.plotly_chart(plot)
        except Exception as e:
            print(e)
###to display graphs

if selected=="Graphs":
      import os
      command="streamlit run ./tsa.py"
      os.system(command)

###text based analysis

if selected=="Text based analysis":
   import streamlit as st
   from textblob import TextBlob
   import pandas as pd
   import altair as alt
   from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

   def convert_to_df(sentiment):
          sentiment_dict = {'polarity': sentiment.polarity, 'subjectivity': sentiment.subjectivity}
          sentiment_df = pd.DataFrame(sentiment_dict.items(), columns=['metric', 'value'])
          return sentiment_df

   def analyze_token_sentiment(docx):
          analyzer = SentimentIntensityAnalyzer()
          pos_list = []
          neg_list = []
          neu_list = []
          for i in docx.split():
              res = analyzer.polarity_scores(i)['compound']
              if res > 0.1:
                 pos_list.append(i)
                 pos_list.append(res)
              elif res <= -0.1:
                 neg_list.append(i)
                 neg_list.append(res)
              else:
                 neu_list.append(i)

          result = {'positives': pos_list, 'negatives': neg_list, 'neutral': neu_list}
          return result

   def main():
         st.title("Sentiment Analysis using Natural Language Processing")
         

         menu = ["Home", "About"]
         choice = st.sidebar.selectbox("Menu", menu)

         if choice == "Home":
             st.subheader("Home")
             with st.form(key='nlpForm'):
                   raw_text = st.text_area("Enter the text you want to analyze here: ")
                   submit_button = st.form_submit_button(label='Analyze')

        # layout
             col1, col2 = st.columns(2)
             if submit_button:
                  with col1:
                    st.info("Results")
                    sentiment = TextBlob(raw_text).sentiment
                    st.write(sentiment)

                # Sentiment
                    if sentiment.polarity > 0:
                       st.markdown("Sentiment:: Positive :smiley: ")
                    elif sentiment.polarity < 0:
                       st.markdown("Sentiment:: Negative :angry: ")
                    else:
                       st.markdown("Sentiment:: Neutral 😐 ")

                # Dataframe
                    result_df = convert_to_df(sentiment)
                    st.dataframe(result_df)

                # Visualization
                    c = alt.Chart(result_df).mark_bar().encode(
                        x='metric',
                        y='value',
                        color='metric')
                    st.altair_chart(c, use_container_width=True)

                  with col2:
                     st.info("Token Sentiment")
                     token_sentiments = analyze_token_sentiment(raw_text)
                     st.write(token_sentiments)

         else:
              st.subheader("About")
              st.write("This website was built by using streamlit")
       

   if __name__ == '__main__':
       main()