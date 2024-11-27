import streamlit as st
import pickle 
import re
import nltk


nltk.download('punkt')
nltk.download('stopwords')

#loading model
knn = pickle.load(open,('knn.pkl','rb'))
tfidf = pickle.load(open,('tfidf.pkl','rb'))

def CleanResume(txt):
    #remove url
    CleanText = re.sub(r'http\S*',' ',txt)      
     # remove @gmail
    CleanText = re.sub(r'@\S*',' ',CleanText)     
    #remove hashtag
    CleanText = re.sub(r'#\S*',' ',CleanText)      
    # Remove RT and CC (whole words only)
    CleanText = re.sub(r'\b(RT|CC)\b',' ',CleanText) 
    #special character 
    Special_text = re.escape(r"""!"#$%'()*+,-./:;<=>?@[\]^_,`{|}~""")
    CleanText = re.sub(r'[%s]' % Special_text,'  ',CleanText)    
     # Replace non-ASCII characters
    CleanText = re.sub(r'\s+',' ',CleanText)                 
    # Replace multiple spaces with a single space
    CleanText = re.sub(r'[^\x00-\x7f]',' ',CleanText)
    return CleanText

def main():
    st.title("Resume Screening App")
    upload_file = st.file_uploader('Upload Resume', type=["txt","pdf"])

    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')
        cleaned_resume = clean_resume([resume_text])
        input_feature = tfidf.transform([cleaned_resume])
        prediction_id = knn.predict(input_feature)[0]
        st.write(prediction_id)
        category_mapping ={
    'Data Science': 6,
    'HR': 12,
    'Advocate': 0,
    'Arts': 1,
    'Web Designing': 24,
    'Mechanical Engineer': 16,
    'Sales': 22,
    'Health and fitness': 14,
    'Civil Engineer': 5,
    'Java Developer': 15,
    'Business Analyst': 4,
    'SAP Developer': 21,
    'Automation Testing': 2,
    'Electrical Engineering': 11,
    'Operations Manager': 18,
    'Python Developer': 20,
    'DevOps Engineer': 8,
    'Network Security Engineer': 17,
    'PMO': 19,
    'Database': 7,
    'Hadoop': 13,
    'ETL Developer': 10,
    'DotNet Developer': 9,
    'Blockchain': 3,
    'Testing': 23
}
category_name = category_mapping.get(prediction_id,"Unknown")
print("Prediction Category: ",prediction_id)
if __name__ == __main__:
    main()