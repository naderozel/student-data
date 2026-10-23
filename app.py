import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber

st.set_page_config(page_title="لوحه المدرسه", page_icon="🏫")

features = ["absence" , 
            "homework", 
            "quiz", 
            "arabic", 
            "math", 
            "english", 
            "science", 
            "social", 
            "art", 
            "sport", 
            "music", 
            "computer", 
            "religion", 
            "behavior", 
            "participation"
]

features_ar = {
    "absence": "نسبه الغياب",   
    "homework": "الواجبات",   
    "quiz": "الاختبارات",   
    "arabic": "العربية",   
    "math": "الرياضيات",   
    "english": "الإنجليزية",   
    "science": " العلوم",   
    "social": "المجتمع",   
    "art": "الفنون",   
    "sport": " الرياضة",   
    "music": "الموسيقى",   
    "computer": " الحاسوب",   
    "religion": "الدين",   
    "behavior": " السلوك",   
    "participation": "المشاركة"   
}

#model
class StudentModel:
    def __init__(self):
        self.weights = None # نسبه تأثير كل فيتشر علي التوقع النهائي 
        self.bias = 0.0
        self.trained = False
    def _normalize(self, x):#0 1
        return (x - self.x_min) / (self.x_max - self.x_min + 1e-8)#0.00000001
    def train(self, x, y, training_num=300, learning_rate=0.3):
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        self.x_min = x.min(axis=0)
        self.x_max = x.max(axis=0)
        
        x = self._normalize(x)
        self.weights = np.zeros(x.shape[1])#[0,0,0,0,0,0]
        self.bias = 0.0

        self.loss_history = []
        for epoch in range(training_num):
            pred = x @ self.weights + self.bias

            error = pred - y
            # mean squared error
            self.loss_history.append(np.mean(error ** 2))
            self.weights -= learning_rate * (x.T @ error) / len(y)
            self.bias -= learning_rate * error.mean()
        self.trained = True
        def predict(self, x):
                x = np.array(x, dtype=float)
                x = (x - self.x_min) / (self.x_max - self.x_min + 1e-8)
                return np.clip(x @ self.weights + self.bias , 0,100)
        
def extract_table_from_pdf(uploaded_file):

    all_rows = [ ]

    header = None

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if not table:
                continue
            if header is None:
                header = table[0]
                all_rows.extend(table[1:])

            else:
                all_rows.extend(table)
    if header is None:
        return None  # Return None if no table is found         
    return pd.DataFrame(all_rows, columns=header)  # Return an empty DataFrame if no table is found    
            
if "model" not in st.session_state:
    st.session_state.model = StudentModel()
if "data" not in st.session_state:
    st.session_state.data = []    
if "trained" not in st.session_state:
    st.session_state.trained = False

st.title("school dashboard")

step1, step2, step3 = st.tabs(["1) Upload TRAINING DATA", "2) Train Model", "3) UPLOAD PDF TO ANALYZE"])

with step1:
    st.header("Upload a pdf containing data from past students")
    train_uploaded= st.file_uploader("Upload a PDF file", type=["pdf"], key="train_pdf")

    if train_uploaded :
        with st.spinner("Extracting data from PDF..."):
            train_df = extract_table_from_pdf(train_uploaded)
            if train_df is None or train_df.empty:
                st.error("No table found in the PDF.")
            else:
                st.success(f"data extracted successfully! {len(train_df)} rows found.")
                st.write("Extracted Data:")
                st.dataframe(train_df.head(8), use_container_width=True)

                st.write("Choose what each column represents:")

                col_options = ["None"] + list(train_df.columns)                 

                train_mapping = {}
                m1, m2, m3, m4, m5 = st.columns(5)

                train_mapping["absence"] = m1.selectbox("absence", train_df.columns, key="absence")
                train_mapping["homework"] = m2.selectbox("homework", train_df.columns, key="homework")
                train_mapping["quiz"] = m3.selectbox("quiz", train_df.columns, key="quiz")
                train_mapping["arabic"] = m4.selectbox("arabic", train_df.columns   , key="arabic")
                train_mapping["math"] = m5.selectbox("math", train_df.columns, key="math")
                train_mapping["english"] = m1.selectbox("english", train_df.columns, key="english")
                train_mapping["science"] = m2.selectbox("science", train_df.columns, key="science")
                train_mapping["social"] = m3.selectbox("social", train_df.columns, key="social")
                train_mapping["art"] = m4.selectbox("art", train_df.columns, key="art")
                train_mapping["sport"] = m5.selectbox("sport", train_df.columns, key="sport")
                train_mapping["music"] = m1.selectbox("music", train_df.columns, key="music")
                train_mapping["computer"] = m2.selectbox("computer", train_df.columns, key="computer")
                train_mapping["religion"] = m3.selectbox("religion", train_df.columns, key="religion")
                train_mapping["behavior"] = m4.selectbox("behavior", train_df.columns, key  ="behavior")
                train_mapping["participation"] = m5.selectbox("participation", train_df.columns , key="participation")

                
                outcome_col = st.selectbox("final result", col_options, key="outcome_col")

                ready = (all(train_mapping[f] != "None" for f in features) and outcome_col != "None")
                   
                if not ready:
                    st.warning("Please map all features and select the final result column.")
                else:
                    if st.button("Use data for training", use_container_width=True):
                        clean = pd.DataFrame() 
                        for f in features:
                            clean[f] = pd.to_numeric(train_df[train_mapping[f]], errors='coerce')
                        clean["outcome"] = pd.to_numeric(train_df[outcome_col], errors='coerce')           
                        clean = clean.dropna()

                        st.session_state.data = (clean[features + ["outcome"]]. values. tolist())

                        st. success(f"{len(clean)} students' data is ready for training")
    if st.session_state.data:
        st.write(f"Number of students' data {len(st.session_state.data)}")

    cols = [features_ar[f] for f in features] + ["Final Result"]
    st.dataframe(pd.DataFrame(st. session_state. data, columns=cols), use_container_width=True, hide_index=True)

    if st.button("Clear data"):
        st.session_state.data = []
        st.session_state. trained = False
        st.session_state.model = StudentModel()
        st.rerun()
with step2:
    st.header("Train the model")
    if len(st.session_state.data) < 8:
        st.warning("You need at least 8 students' data to train the model.")
    else:
        if st.button("training", use_container_width=True, type="primary"):
            data = st.session_state.data
            x = [d[:6] for d in data]
            y = [d[6] for d in data]

            with st.spinner("Training the model..."):
                st.session_state.model.train(x, y)
                st.session_state.trained = True
            st.success("Model trained successfully!")
            st.snow()

        if st.session_state.trained:
            model = st.session_state.model
        else:
            st.info("train the model first to see the weights and bias.")     

with step3:
    st.header("Upload your test pdf")
    if not st.session_state.trained:
        st.info("Train your model first")
    else:
        model = st.session_state.model
        uploaded = st.file_uploader("Choose your pdf", type=['pdf' ])

        if uploaded:
            with st.spinner("Extracting your table"):
                df = extract_table_from_pdf(uploaded)
            if df is None or df.empty:
                st.error("No table found in the PDF.")
            else:
                st.success(f"There is a {df.shape[0]} rows, {df.shape[1]} columns")
                st.write("Extracted Data:")
                st.dataframe(df.head(8), use_container_width=True)

                st.write("Choose what each column represents:")
                col_options = ["None"] + list(df.columns)

                mapping = {}

                m1, m2, m3, m4, m5 = st.columns(5)
                name_col = m1.selectbox("Student Name", col_options)

                mapping["absence"] = m2.selectbox("absence", col_options , key="map_absence")
                mapping["homework"] = m3.selectbox("homework", col_options , key="map_homework")
                mapping["quiz"] = m4.selectbox("quiz", col_options , key="map_quiz")
                mapping["arabic"] = m5.selectbox("arabic", col_options , key="map_arabic")
                mapping["math"] = m1.selectbox("math", col_options , key="map_math")
                mapping["english"] = m2.selectbox("english", col_options , key="map_english")
                mapping["science"] = m3.selectbox("science", col_options , key="map_science")
                mapping["social"] = m4.selectbox("social", col_options , key="map_social")
                mapping["art"] = m5.selectbox("art", col_options , key="map_art")
                mapping["sport"] = m1.selectbox("sport", col_options , key="map_sport")
                mapping["music"] = m2.selectbox("music", col_options , key="map_music")
                mapping["computer"] = m3.selectbox("computer", col_options , key="map_computer")
                mapping["religion"] = m4.selectbox("religion", col_options , key="map_religion")
                mapping["behavior"] = m5.selectbox("behavior", col_options , key="map_behavior")
                mapping["participation"] = m1.selectbox("participation", col_options , key="map_participation") 

                ready = all(mapping[f] != "None" for f in features)

                if not ready:
                    st.warning("Please choose all features.")
                else:
                    if st.button("start analyzing" , use_container_width=True):
                        work = pd.DataFrame()
                        if name_col != "None":
                            work["Student Name"] = df[name_col]
                        for f in features:
                            work[features_ar[f]] = pd.to_numeric(df[mapping[f]], errors='coerce')
                        work = work.dropna(subset=features)

                        work["pre"] = model.predict_all(work[features].values).round(0)
                        work["Status"] = np.where(work["Pre"] >= 50, "Success", "Faild")

                        n = len(work)
                        n_pass = (work["pre"] >= 50).sum()
                        n_fail = n - n_pass

                        n_excellent = (work["pre"] >= 80).sum()

                        st.divider()

                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("Total Students", n)
                        k2.metric("Pass Students", n_pass)
                        k3.metric("Fail Students", n_fail)
                        k4.metric("Excellent Students", n_excellent)

                        sub_av = {features_ar[s]: work[s].mean() for s in ["arabic", "math", "english", "science", "social", "art", "sport", "music", "computer", "religion", "behavior", "participation"]}
                        w_sub = min(sub_av, key=sub_av.get)
                        st.write(f"faild students")
                        at_risk = work[work["pre"] < 50].sort_values("pre")

                        if len(at_risk) == 0:
                            st.info("No students")
                        else:
                            st.dataframe(at_risk, use_container_width=True, hide_index=True)

                            st.write("Result")
                            st.dataframe(work.sort_values("Pre"), use_container_width=True, hide_index=True) 