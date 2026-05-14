import streamlit as st
import pandas as pd
import joblib
sentiment=joblib.load("review.pkl")
lang=joblib.load("langdetect.pkl")
spam=joblib.load("spam_classifier")
house=joblib.load("bengaluru_house_predict")
recom=joblib.load("recommendation")
import requests
# st.title("second streamlit")
# st.header("this is header")
# st.subheader("this is subheader")
# st.write("this is write")
# st.text("this is text")
# st.warning("this is warning")
# st.error("this is error")
# st.success("this is success")
# st.markdown("this is markdoen")

# st.markdown("this is link[google]('www.google.com')")
# fileup=st.file_uploader("choose a file",type=["text","csv"])
# if fileup:
#     import pandas as pd 
#     df=pd.read_csv(fileup)
#     st.dataframe(df)
# num=st.number_input("enter the num")
# # bt=st.button("even/odd")
# # if bt:
# #     if(num%2==0):
# #         st.write("EVEN")
# #     else:
# #         st.write("ODD")    

# slider_age=st.slider("select your age",min_value=10,max_value=80)
# if st.button("even/odd"):
#     st.write(slider_age)
#     if(slider_age%2==0):
#         st.write("EVEN")
#     else:
#         st.write("ODD") 

# col1,col2=st.columns(2)
# with col1:
#     name=st.text_input("give name")
#     st.write(name)
# with col2:
#     dob=st.date_input("DOB") 
#     st.write(dob)
# st.success("hello")
# # st.sidebar.image("D:\datasets\sumitphoto.jpeg")  

# st.sidebar.button("SPAM Classifier")     
# st.sidebar.button("Language Detection") 
# st.sidebar.button("Food Review Sentiment")
# CSS for crop
st.markdown("""
<style>
[data-testid="stImage"] img {
    width: 100% !important;
    height: 100px !important;  
    object-fit: fill !important; 
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

col1,col2,col3=st.columns([20,20,20])
with col1:
    st.image("D:/datasets/sakuraimage.jpg")

with col2:
    st.image("D:/datasets/machine.jpg")
with col3:
    st.image("D:/datasets/sakuraimage.jpg")
# 🧠 Title
st.title("🤖 Machine Learning Projects Hub")
tab1, tab2, tab3, tab4 ,tab5= st.tabs(["🍔 Food Review","📩 Spam Classifier","🌐 Language Detection","🏠 House Prediction","Movie recommendation"])
with tab1:
    msg=st.text_input("enter the message",key="msg_tab1")
    if st.button("prediction",key="tab1"):
        pred=sentiment.predict([msg])
        if pred[0]==0:
            st.text("Negative")
        else:
            st.text("Positive")
            st.balloons()
    uploaded_file1 = st.file_uploader("Upload CSV for batch prediction", type=["csv"])

    if uploaded_file1 is not None:
        df = pd.read_csv(uploaded_file1)
        st.write("📊 Uploaded Data", df.head())

    if st.button("Predict on Dataset"):
        predictions = sentiment.predict(df["review"])   # 👈 column name check karna

        df["Prediction"] = predictions
        df["Prediction"] = df["Prediction"].map({0: "Negative", 1: "Positive"})

        st.success("✅ Prediction Done")
        st.dataframe(df)



with tab2:
    msg2=st.text_input("enter the message",key="msg_tab2")
    if st.button("prediction",key="tab2"):
        pred2=spam.predict([msg2])
        if pred2[0]==0:
            st.text("Spam")
        else:
            st.text("ham")
            st.balloons()
    # uploaded_file2=st.file_uploader("choose a file",type=["csv","txt"])
    
    
with tab3:
    msg1=st.text_input("enter the message",key="msg_tab3")
    if st.button("DETECT LANGUAGE",key="tab3"):
        pred1=lang.predict([msg1])
        language=pred1[0]
        languages=["ENGLISH", "URDU", "PUNJABI", "SANSKRIT", "HINDI"]
        if language in languages:
            st.success(language)
        else:
            st.error("Unsupported language")      

    # uploaded_file2=st.file_uploader("choose a file",type=["csv","txt"])
with tab4:
    area1=st.selectbox("AREA TYPE",options=["Super built-up Area","Plot Area",'Built-up Area','Carpet Area'])
    loc1=st.selectbox("LOCATION",options=['Electronic City Phase II','others','Uttarahalli','Kothanur',
       'Whitefield','Rajaji Nagar','Marathahalli','7th Phase JP Nagar',
       'Sarjapur','Raja Rajeshwari Nagar','Kengeri','Thanisandra',
       'Bellandur','Electronic City','Hebbal','Kanakpura Road',
       'Electronics City Phase 1','Sarjapur  Road','Yelahanka',
       'KR Puram','Begur Road','Varthur','Haralur Road','Hennur Road',
       'Kasavanhalli','Yeshwanthpur','Chandapura','Nagarbhavi',
       'Ramamurthy Nagar','Malleshwaram','Akshaya Nagar','Hormavu',
       'Hulimavu','Hosa Road','Koramangala','Old Madras Road',
       'Kaggadasapura','Jakkur','JP Nagar','Harlur',
       'Bannerghatta Road','8th Phase JP Nagar','Hoodi','Banashankari'])
    soc1=st.selectbox("SOCIETY",options=['Others', 'Prityel', 'GrrvaGr', 'Soitya ', 'Itelaa ', 'ViistLa',
       'PrarePa', 'Dieldli', 'Rosha I', 'RothaVa', 'Dhalsh ', 'SNnia E',
       'Bhmesy ', 'IBityin', 'Sryalan', 'GMown E', 'Prarkun', 'PrityTr',
       'JRrnauv', 'PhestOn', 'Prtanha', 'Prlla C', 'GoAirej', 'SNity S',
       'Prtates', 'MenueNo', 'PuandHi', 'Adeatlm', 'Prncyrn', 'DLhtsnd',
       'Puachal', 'Soresea', 'Bhe 2ko'])
    bhk1=st.selectbox("Select the BHK",options=[2,4,3,6,1,8,7,5])
    squareft1=st.slider("Total Square feet",min_value=1,max_value=346786)
    bath1=st.slider("Bathroom",min_value=1,max_value=8)
    balcony1=st.slider("Balcony",min_value=0,max_value=3)
    if st.button("Bengaluru House Prediction",key="tab4"):
        msg4=pd.DataFrame([{"area_type":area1,"location":loc1,"size":bhk1,"society":soc1,"total_sqft":squareft1,"bath":bath1,"balcony": balcony1}])
        pred4=house.predict(msg4)
        price=pred4[0]
        st.success(f"{price.round(3)} lakh")
 
with tab5:
    movie1=st.selectbox("MOVIE NAME",options=recom["movies_list"]).lower()
    if st.button("prediction",key="tab5"):
        model = recom["model"]
        vectors = recom["vectors"]
        df = recom["df5"]

# Step 1: movie index nikalo
        idx = df[df["name"] == movie1].index[0]

# Step 2: vector nikalo
        movie_vector = vectors[idx]

# Step 3: kneighbors
        distances, indexes = model.kneighbors([movie_vector], n_neighbors=5)
        cols = st.columns(5) 
        l=1
        for i in indexes[0][1:]:
            with cols[l]:
                st.write(f"{df.loc[i]["name"]}")
                api=f"http://www.omdbapi.com/?i={df.loc[i]["movie_id"]}&apikey=93c66664"
                resp=requests.get(api)
                
                if (resp.json()["Poster"]=="N/A"):
                    st.write("POSTER NOT AVAILABLE")
                else:
                    st.image(resp.json()["Poster"])
                l=l+1 
st.sidebar.title("📞 Contact Info")

# st.image("D:/datasets/sakuraimage.jpg",use_container_width=True)
st.sidebar.image("D:/datasets/contactus.jpg")
# st.image("sakuraimage.jpg",)


st.sidebar.markdown("""
📱 **Phone:** 9457746751
📧 **Email:** rawatsumit231@gmail.com  
""")


