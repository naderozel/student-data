import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber #pip install pdfplumber

st.set_page_config(page_title="School Predictor", page_icon="🏫")

features = ["attendace" , "homework", "quiz", "arabic", "math", "english"]
features_ar = {
    "attendace": "نسبه الحضور",
    "homework": "درجه الواجبات",
    "quiz": "درجه الاختبار",
    "arabic": "عربي",
    "math": "رياضيات",
    "english": "انجليزي"
}

# Model
class SimpleModel:
    # function (method) -> auto call
    def __init__ (self) :
        self.weigths = None # نسبه تأثير كل فيتشر علي التوقع النهائي
        self.bais = 0.0
        self.trained = False
    def _normalize(self, X): #0 1
        return (X - self.x_min) / (self.x_max - self.x_min + 1e-8) #0.000000001

    def train(self, X, y, training_num=300, lr = 0.3):
        X = np.array(X , dtype=float)
        y = np.array(y , dtype=float)
        self.x_min = X.min(axis=0)
        self.x_max = X.max(axis=0)

        X = self._normalize(X)
        self.weigths = np.zeros(X.shape[1]) #[0,0,0,0,0,0]
        self.bais = 0.0

        self.loss_history = []
        for epoch in range(training_num):
            pred = X @ self.weigths + self.bais

            error = pred - y
            # mean squared error
            self.loss_history.append(np.mean(error**2))
            self.weigths -= lr * (X.T @ error) / len(y)
            self.bais -= lr * error.mean()

        self.trained = True

    def predict_all (self, X):
        X = np.array(X, dtype=float)
        X = (X - self.x_min) / (self.x_max - self.x_min + 1e-8)

        return np.clip(X @ self.weigths + self.bais , 0, 100) 

def extract_table_from_pdf(uploaded_file):
    all_rows = []
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
            return None
        
        return pd.DataFrame(all_rows, columns=header)
            

if "model" not in st.session_state:
    st.session_state.model = SimpleModel()
if "data" not in st.session_state:
    st.session_state.data = []
if "trained" not in st.session_state:
    st.session_state.trained = False

st.title("School Dashboard")

step1 , step2, step3 = st.tabs(["1) Upload the training data pdf", "2) Train your model", "3) Upload pdf to analyze"])

with step1 :
    st.header("Uplaod a pdf containing data from past students")

    train_uploaded = st.file_uploader("Choose pdf file", type=["pdf"], key="train_pdf")

    if train_uploaded:
        with st.spinner("Reading the table from pdf..."):
            train_df = extract_table_from_pdf(train_uploaded)

        if train_df is None or train_df.empty:
            st.error("No table was found in the pdf")
        else:
            st.success(f"Table extracted successfuly with {len(train_df)} rows was found")
            st.write("Extracted Data Preview")
            st.dataframe(train_df.head(8), use_container_width=True) 

            st.write("Choose what each column represents")
            col_options = ["None"] + list(train_df.columns)

            train_mapping = {}

            m1, m2, m3 = st.columns(3)

            train_mapping["attendace"] = m1.selectbox("Attendacne", col_options, key="tmap_att")
            train_mapping["homework"] = m2.selectbox("Homework", col_options, key="tmap_hw")
            train_mapping["quiz"] = m3.selectbox("quiz", col_options, key="tmap_qz")

            m4, m5, m6 = st.columns(3)

            train_mapping["arabic"] = m4.selectbox("arabic", col_options, key="tmap_ar")
            train_mapping["math"] = m5.selectbox("math", col_options, key="tmap_ma")
            train_mapping["english"] = m6.selectbox("english", col_options, key="tmap_en")

            outcome_col = st.selectbox("Final Result", col_options, key="tmap_out")

            ready = (all(train_mapping[f] != "None" for f in features) and outcome_col != "None")
            if not ready:
                st.warning("Choose all required columns")
            else:
                if st.button("Use data for training", use_container_width=True):
                    clean = pd.DataFrame()
                    for f in features: #NaN
                        clean[f] = pd.to_numeric(train_df[train_mapping[f]], errors="coerce")
                    clean['outcome'] = pd.to_numeric(train_df[outcome_col], errors="coerce")

                    clean = clean.dropna()

                    st.session_state.data = (clean[features + ["outcome"]].values.tolist())

                    st.success(f"{len(clean)} students' data is ready for training")
    if st.session_state.data:
        st.write(f"Number of students' data {len(st.session_state.data)}")

        cols = [features_ar[f] for f in features] + ["Final Result"]
        st.dataframe(pd.DataFrame(st.session_state.data, columns=cols), use_container_width=True, hide_index=True)

        if st.button("Clear data"):
            st.session_state.data = []
            st.session_state.trained = False
            st.session_state.model = SimpleModel()
            st.rerun() 

with step2:
    st.header("Train your model")
    if len(st.session_state.data) < 8 :
        st.warning("You need more than 8 students")
    else:
        if st.button("Training", use_container_width=True, type="primary"):
            data = st.session_state.data
            x = [d[:6] for d in data]
            y = [d[6] for d in data] 

            with st.spinner("The model is learning"):
                st.session_state.model.train(x,y)
                st.session_state.trained = True

            st.success("Model trained successfuly")
            st.snow()
            # st.snow()

        if st.session_state.trained:
            model = st.session_state.model
        else:
            st.info("Train your model first so you can go to step 3")

with step3:
    st.header("Upload your test pdf")
    if not st.session_state.trained:
        st.info("Train your model first")
    else:
        model = st.session_state.model
        uploaded = st.file_uploader("Choose your pdf", type=['pdf'])

        if uploaded:
            with st.spinner("Extracting your table"):
                df = extract_table_from_pdf(uploaded)

            if df is None or df.empty:
                st.error("Make sure there is a table in your pdf")
            else:
                st.success(f"There is a {len(df)} rows")
                st.write("Extracted data")
                st.dataframe(df.head(8), use_container_width=True)

                st.write("Choose each column")

                col_options = ["None"] + list(df.columns)

                mapping = {}

                m1, m2 , m3 = st.columns(3)
                name_col = m1.selectbox("Student name", col_options)

                mapping["attendace"] = m2.selectbox("attendace", col_options, key= "map_att")
                mapping["homework"] = m3.selectbox("homework", col_options, key= "map_hw")
                
                m4, m5 , m6 = st.columns(3)
                mapping["quiz"] = m4.selectbox("quiz", col_options, key= "map_qz")
                mapping["arabic"] = m5.selectbox("arabic", col_options, key= "map_ar")
                mapping["math"] = m6.selectbox("math", col_options, key= "map_ma")

                mapping["english"] = st.selectbox("english", col_options, key= "map_en")

                ready = all(mapping[f] != "None" for f in features)

                if not ready :
                    st.warning("Choose all your columns")
                else:
                    if st.button("Strat analysing" , use_container_width=True, type="primary"):
                        work = pd.DataFrame()
                        if name_col != "None":
                            work["Student Name"] = df[name_col]

                        for f in features:
                            work[f] = pd.to_numeric(df[mapping[f]], errors="coerce")

                        work =work.dropna(subset=features)

                        work["Pre"] = model.predict_all(work[features].values).round(0)
                        work["Status"] = np.where(work["Pre"] >= 50, "Success", "Faild")

                        n = len(work)

                        n_pass = int((work["Pre"] >= 50).sum())
                        n_fail = n - n_pass

                        n_ex = int((work["Pre"] >= 80).sum())

                        st.divider()

                        k1,k2,k3,k4 = st.columns(4)

                        k1.metric("Total students" , n)
                        k2.metric("Pass students" , n_pass)
                        k3.metric("Fail students" , n_fail)
                        k4.metric("Excellent students" , n_ex)

                        sub_av = {features_ar[s]: work[s].mean() for s in ["arabic", "math", "english"]}

                        w_sub = min(sub_av, key=sub_av.get)

                        st.write(f"Weakest subject: {w_sub}")

                        st.bar_chart(pd.Series(sub_av))

                        st.write("Faild student")
                        at_risk = work[work["Pre"] < 50].sort_values("Pre")

                        if len(at_risk) == 0:
                            st.info("No students")
                        else:
                            st.dataframe(at_risk, use_container_width=True, hide_index=True)

                        st.write("Result")
                        st.dataframe(work.sort_values("Pre"), use_container_width=True, hide_index=True)
