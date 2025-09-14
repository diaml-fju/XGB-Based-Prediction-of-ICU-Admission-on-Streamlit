import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap

# ------------------------- 分頁切換 -------------------------
st.sidebar.title("Explainable ML for Personalized Assessment of the Need for ICU in Myasthenia Gravis (PredMGICU)")
st.sidebar.markdown("""
Che-Cheng Chang, Kuan-Yu Lin, Jiann
Horng Yeh, Hou-Chang Chiu, Tzu-Chi Liu and Chi
Jie Lu""")
st.sidebar.title("Instructions for Feature Input")
st.sidebar.markdown("""
Please input your data as follows:""")
st.sidebar.markdown("""
                    (1)Selection of the phenotype of MG""")
st.sidebar.markdown("""(2)Enter the clinical, inflammation markers as bellowing""") 
st.sidebar.title("MG phenotype selection")
model_choice = st.sidebar.selectbox("",[
    "EOMG",
    "LOMG",
    "Thymoma",
    "Non-Thymoma"
])
st.sidebar.title("Important varialbes input")
#tab1, tab2, tab3, tab4 = st.tabs(["EOMG", "LOMG", "Thymoma", "Non-Thymoma"])


# ------------------------- 共用函數：預測 + SHAP -------------------------
def predict_and_explain(model, x_train, input_df, model_name):
    import shap
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import streamlit as st
    import xgboost as xgb
    st.subheader("Predict of Outcomes")

    try:
        # 特徵對齊
        model_feature_names = model.get_booster().feature_names
        input_df = input_df[model_feature_names]
        background = x_train[model_feature_names]

        # 預測
        #proba = model.predict_proba(input_df)[0]
        #pred_class = int(np.argmax(proba))
        

        #if pred_class == 1:
        #    st.error("Positive risk of ICU admission")
        #else:
        #    st.success("Negative risk of ICU admission")
        proba = model.predict_proba(input_df)[0]
        positive_prob = proba[1]

        adaptive_thresholds = {

            "EOMG":0.14298505,
            "LOMG":0.5117961,
            "Thymoma":0.41806757,
            "Non-Thymoma":0.10304403
        }
        threshold = adaptive_thresholds[model_choice]
        if positive_prob >= threshold:
            st.error(f"Positive risk of ICU admission (Prob = {positive_prob:.4f})")
        else:
            st.success(f"Negative risk of ICU admission (Prob = {positive_prob:.4f})")
        # SHAP 解釋
        explainer = shap.TreeExplainer(model, data=background,model_output="probability", feature_perturbation="interventional")
        shap_values = explainer.shap_values(input_df)
        # ✅ 防止 index 錯誤
        shap_val = shap_values[0]
        #st.write("Shap_values",shap_values)
        #st.write("SHAP",shap_val)
        base_val = explainer.expected_value
        st.subheader("SHAP based personalized explanation")
        fig = plt.figure()
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_val,
                base_values=base_val,
                data=input_df.values[0],
                feature_names=input_df.columns.tolist()
            ),
            
            show=False
        )
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error：{e}")

# ✅ 定義通用二元選單函式
def binary_radio(label,key= None,index=None):
    return st.radio(
        label,
        options=[1, 0],
        format_func=lambda x: f"Yes" if x == 1 else f"No",
        index=index,
        key=key
    )

def binary_radio_Thymic(label,key= None,index=None):
    return st.radio(
        label,
        options=[1, 0],
        format_func=lambda x: f"Absence" if x == 0 else f"Presence",
        key=key,
        index=index
    )



# ------------------------- 模型 A -------------------------
def run_model_a_page():
    st.title("Intensive care risk prediction result")
    st.markdown("""We 
provide detailed guidance through 
step-by-step instructions. Users can 
download the file below:""")
    with open("Test.pdf", "rb") as f:
        st.download_button(
            label="📥 Download of user-guide",
            data=f,
            file_name="Test.pdf",
            mime="application/pdf"
        )

    # 模型 & 資料（你之後替換正確路徑）
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(r"MG_ICU_SHAP_XGB_EOMG.json")
    x = pd.read_csv(r"MG_ICU_SHAP_Model_Data_SubGroup2_EOMG.csv")
    x_train = x.drop(columns=[ "Y","MGFA clinical classification"])
    # 輸入變數
    # ➤ Clinical variables
    with st.sidebar.expander("Clinical variables", expanded=True):
        #Age = st.number_input("Age at onset (year)", 50,disabled=True)
        Gender = st.radio(
        "Gender",
        options=[(1, "Male"), (2, "Female")],
        format_func=lambda x: x[1],
        key="EOMG_Gender",
        index=1
        )
        Gender = Gender[0]  
        Disease_duration= st.number_input("Disease duration (month)", min_value=0.01, value=48.0, key="EOMG_Disease_duration")
        BMI = st.number_input("BMI", min_value=0.01, value=17.89)

    #MGFA
    # ➤ Corticosteroid variables
    with st.sidebar.expander("Treatment related variables", expanded=False):
    
        Prednisolone = st.number_input("Prednisolone daily dose before admission (mg)", min_value=0.0, value=10.0)
        Immunosuppressant = st.radio(
        "Immunosuppressant at admission", 
        options=[(1, "Azathioprine"), (2, "Calcineurin"), (3, "Mycophenolate"), (4, "Quinine"),(0, "None of above")], 
        format_func=lambda x: x[1],
        key="EOMG_Immuno",index = 4
        )
        Immunosuppressant = Immunosuppressant[0]
    # ➤ Thymic pathology
    with st.sidebar.expander("Thymic pathology variables", expanded=False):
    
        Thymoma = binary_radio_Thymic("Thymoma", key="EOMG_Thymoma",index=0)
        Thymic = binary_radio_Thymic("Thymic hyperplasia", key="EOMG_Thymic",index=1)
        Thymectomy = binary_radio("Thymectomy", key="EOMG_Thymectomy",index=0)
    
    # ➤ Serology
    with st.sidebar.expander("Serology of autoantibody", expanded=False):

        Anti_AChR = binary_radio("Anti-AChR", key="EOMG_Anti_AChR",index=0)
        Anti_MuSK = binary_radio("Anti-MuSK", key="EOMG_Anti_MuSK",index=1)
        dSN = binary_radio("dSN", key="EOMG_dSN",index=1)

    # ➤ Comorbidity
    with st.sidebar.expander("Comorbidity variables", expanded=False):
        Infection = binary_radio("Infection at admission", key="EOMG_Infection",index=0)
        Thyroid = binary_radio("Thyroid disease", key="EOMG_Thyroid",index=1)
        Diabetes = binary_radio("Diabetes", key="EOMG_Diabetes",index=1)
        Hypertension = binary_radio("Hypertension", key="EOMG_Hypertension",index=1)
        Auto = binary_radio("Autoimmune disease", key="EOMG_Auto",index=1)
        ASCVD = binary_radio("ASCVD", key="EOMG_ASCVD",index=0)
        Chronic = binary_radio("Chronic lung disease", key="EOMG_Chronic",index=0)
        Good = binary_radio("Good syndrome", key="EOMG_Good",index=1)

    # ➤ Inflammation
    with st.sidebar.expander("Systemic inflammation markers variables", expanded=False):
        NLR = st.number_input("NLR", min_value=0.01, key="EOMG_NLR",value=4.286549708)
        PLR = st.number_input("PLR", min_value=0.01, key="EOMG_PLR",value=237.6115728)
        LMR = st.number_input("LMR", min_value=0.01, key="EOMG_LMR",value=2.23880597)
        SII = st.number_input("SII", min_value=0.01, key="EOMG_SII",value=1654608.2)
    
    # 建立 dict（易於維護）
    input_dict = {
    "Gender": Gender,
    "BMI": BMI,
    "Infection at admission": Infection,
    "Thyroid disease": Thyroid,
    "Autoimmune disease": Auto, 
    "Diabetes": Diabetes,
    "Hypertension": Hypertension,
    "ASCVD": ASCVD,
    "Chronic lung disease": Chronic,
    "Good syndrome": Good,
    "Disease duration (month)": Disease_duration,
    "Prednisolone daily dose before admission": Prednisolone,
    "Immunosuppressant at admission": Immunosuppressant,
    "Anti-MuSK": Anti_MuSK,
    "Anti-AChR": Anti_AChR,
    "dSN": dSN,
    "Thymoma": Thymoma,
    "Thymic hyperplasia": Thymic,
    "Thymectomy": Thymectomy,
    "NLR": NLR,
    "PLR": PLR,
    "LMR": LMR,
    "SII": SII
}

    


    if st.sidebar.button("Analysis"):
        # 用 input_dict 建立 DataFrame
       # 建立 DataFrame（按照 x_train 的欄位順序）
        input_df = pd.DataFrame([[input_dict[col] for col in x_train.columns]], columns=x_train.columns)
        # 印出模型實際特徵
        model_feature_names = model.get_booster().feature_names
        

        # 僅保留模型實際特徵
        input_df = input_df[model_feature_names]
        
        predict_and_explain(model, x_train, input_df, "model A")

# ------------------------- 模型 B -------------------------

def run_model_b_page():
    st.title("Intensive care risk prediction result")
    st.markdown("""We 
provide detailed guidance through 
step-by-step instructions. Users can 
download the file below:""")
    with open("Test.pdf", "rb") as f:
        st.download_button(
            label="📥 Download of user-guide",
            data=f,
            file_name="Test.pdf",
            mime="application/pdf"
        )
    # 模型 & 資料（你之後替換正確路徑）
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(r"MG_ICU_SHAP_XGB_LOMG.json")
    x = pd.read_csv(r"MG_ICU_SHAP_Model_Data_SubGroup2_LOMG.csv")
    x_train = x.drop(columns=[ "Y","MGFA clinical classification"])
    # ➤ Clinical variables
    with st.sidebar.expander("Clinical variables", expanded=True):
        #Age = st.number_input("Age at onset (year)", 50, disabled=True, key="LOMG_Age")
        Gender = st.radio(
            "Gender",
            options=[(1, "Male"), (2, "Female")],
            format_func=lambda x: x[1],index=1
        )
        Gender = Gender[0]
        Disease_duration = st.number_input("Disease duration (month)", min_value=0.01, value=108.0, key="LOMG_Disease_duration")
        BMI = st.number_input("BMI", min_value=0.01, value=18.63, key="LOMG_BMI")

    # ➤ Corticosteroid variables
    with st.sidebar.expander("Treatment related variables", expanded=False):
        Prednisolone = st.number_input("Prednisolone daily dose before admission (mg)", min_value=0.0, value=0.0, key="LOMG_Prednisolone")
        Immunosuppressant = st.radio(
        "Immunosuppressant at admission", 
        options=[(1, "Azathioprine"), (2, "Calcineurin"), (3, "Mycophenolate"), (4, "Quinine"),(0, "None of above")], 
        format_func=lambda x: x[1],
        key="LOMG_Immuno",index=4
        )
        Immunosuppressant = Immunosuppressant[0]

    # ➤ Thymic pathology
    with st.sidebar.expander("Thymic pathology variables", expanded=False):
        Thymoma = binary_radio_Thymic("Thymoma", key="LOMG_Thymoma",index=0)
        Thymic = binary_radio_Thymic("Thymic hyperplasia", key="LOMG_Thymic",index=0)
        Thymectomy = binary_radio("Thymectomy", key="LOMG_Thymectomy",index=0)

    # ➤ Serology
    with st.sidebar.expander("Serology of autoantibody", expanded=False):
        Anti_AChR = binary_radio("Anti-AChR", key="LOMG_Anti_AChR",index=0)
        Anti_MuSK = binary_radio("Anti-MuSK", key="LOMG_Anti_MuSK",index=1)
        dSN = binary_radio("dSN", key="LOMG_dSN",index=1)

    # ➤ Comorbidity
    with st.sidebar.expander("Comorbidity variables", expanded=False):
        Infection = binary_radio("Infection at admission", key="LOMG_Infection",index=1)
        Thyroid = binary_radio("Thyroid disease", key="LOMG_Thyroid",index=1)
        Diabetes = binary_radio("Diabetes", key="LOMG_Diabetes",index=1)
        Hypertension = binary_radio("Hypertension", key="LOMG_Hypertension",index=0)
        Auto = binary_radio("Autoimmune disease", key="LOMG_Auto",index=1)
        ASCVD = binary_radio("ASCVD", key="LOMG_ASCVD",index=0)
        Chronic = binary_radio("Chronic lung disease", key="LOMG_Chronic",index=1)
        Good = binary_radio("Good syndrome", key="LOMG_Good",index=1)

    # ➤ Inflammation
    with st.sidebar.expander("Systemic inflammation markers variables", expanded=False):
        NLR = st.number_input("NLR", min_value=0.01,  key="LOMG_NLR",value=11.41791045)
        PLR = st.number_input("PLR", min_value=0.01,  key="LOMG_PLR",value=779.8507463)
        LMR = st.number_input("LMR", min_value=0.01,  key="LOMG_LMR",value=3.909090909)
        SII = st.number_input("SII", min_value=0.01,  key="LOMG_SII",value=2386343.3
)

    
    # 建立 dict（易於維護）
    input_dict = {
    "Gender": Gender,
    "BMI": BMI,
    "Infection at admission": Infection,
    "Thyroid disease": Thyroid,
    "Autoimmune disease": Auto, 
    "Diabetes": Diabetes,
    "Hypertension": Hypertension,
    "ASCVD": ASCVD,
    "Chronic lung disease": Chronic,
    "Good syndrome": Good,
    "Disease duration (month)": Disease_duration,
    "MGFA clinical classification": 0,
    "Prednisolone daily dose before admission": Prednisolone,
    "Immunosuppressant at admission": Immunosuppressant,
    "Anti-MuSK": Anti_MuSK,
    "Anti-AChR": Anti_AChR,
    "dSN": dSN,
    "Thymoma": Thymoma,
    "Thymic hyperplasia": Thymic,
    "Thymectomy": Thymectomy,
    "NLR": NLR,
    "PLR": PLR,
    "LMR": LMR,
    "SII": SII
}

    if st.sidebar.button("Analysis"):
        # 用 input_dict 建立 DataFrame
       # 建立 DataFrame（按照 x_train 的欄位順序）
        input_df = pd.DataFrame([[input_dict[col] for col in x_train.columns]], columns=x_train.columns)
        # 印出模型實際特徵
        model_feature_names = model.get_booster().feature_names
        

        # 僅保留模型實際特徵
        input_df = input_df[model_feature_names]
        
        predict_and_explain(model, x_train, input_df, "Model B")

def run_model_c_page():
    st.title("Intensive care risk prediction result")
    st.markdown("""We 
provide detailed guidance through 
step-by-step instructions. Users can 
download the file below:""")
    with open("Test.pdf", "rb") as f:
        st.download_button(
            label="📥 Download of user-guide",
            data=f,
            file_name="Test.pdf",
            mime="application/pdf"
        )
    # 模型 & 資料（你之後替換正確路徑）
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(r"MG_ICU_SHAP_XGB_Thymoma.json")
    x = pd.read_csv(r"MG_ICU_SHAP_Model_Data_SubGroup1_Thymoma_Yes.csv")
    x_train = x.drop(columns=[ "Y"])
    # 輸入變數
    # ➤ Clinical variables
    with st.sidebar.expander("Clinical variables", expanded=True):
        Age= st.number_input("Age at onset (year)", min_value=0.00000001, value=44.0, key="Thymoma_Age")
        Gender = st.radio(
        "Gender",
        options=[(1, "Male"), (2, "Female")],
        format_func=lambda x: x[1],index=1
        )
        Gender = Gender[0]
        Disease_duration= st.number_input("Disease duration (month)", min_value=0.00000001, value=48.0, key="Thymoma_Disease_duration")
        BMI = st.number_input("BMI", min_value=0.00000001, value=17.89, key="Thymoma_BMI")
    #MGFA
    # ➤ Corticosteroid variables
    with st.sidebar.expander("Treatment related variables", expanded=False):
        Prednisolone = st.number_input("Prednisolone daily dose before admission (mg)", min_value=0.0, value=10.0, key="Thymoma_Prednisolone")
        Immunosuppressant = st.radio(
        "Immunosuppressant at admission", 
        options=[(1, "Azathioprine"), (2, "Calcineurin"), (3, "Mycophenolate"), (4, "Quinine"),(0, "None of above")], 
        format_func=lambda x: x[1],
        key="EOMG_Immuno",index = 4
        )
        Immunosuppressant = Immunosuppressant[0]
    # ➤ Thymic pathology
    with st.sidebar.expander("Thymic pathology variables", expanded=False):

        Recurrent_thymoma = binary_radio("Recurrent thymoma", key="Thymoma_Recurrent_thymoma",index = 1)
        Invasive_thymoma = binary_radio("Invasive thymoma", key="Thymoma_Invasive_thymoma",index = 0)
    # ➤ Comorbidity variables
    with st.sidebar.expander("Comorbidity variables", expanded=False):
    
        Infection = binary_radio("Infection at admission", key="Thymoma_Infection",index = 0)
        Thyroid = binary_radio("Thyroid disease", key="Thymoma_Thyroid",index = 1)
        Diabetes = binary_radio("Diabetes", key="Thymoma_Diabetes",index = 1)
        Hypertension = binary_radio("Hypertension", key="Thymoma_Hypertension",index = 1)
        Auto = binary_radio("Autoimmune disease", key="Thymoma_Auto",index = 1)
        ASCVD = binary_radio("ASCVD", key="Thymoma_ASCVD",index = 0)
        Chronic = binary_radio("Chronic lung disease", key="Thymoma_Chronic",index = 0)
        Good = binary_radio("Good syndrome", key="Thymoma_Good",index = 1)

    # ➤ Systemic inflammation markers profile
    with st.sidebar.expander("Systemic inflammation markers profile variables", expanded=False):
        #WBC
        NLR = st.number_input("NLR", min_value=0.00000001, value=4.286549708
, key="Thymoma_NLR")
        PLR = st.number_input("PLR", min_value=0.00000001, value=237.6115728
, key="Thymoma_PLR")
        LMR = st.number_input("LMR", min_value=0.00000001, value=2.23880597
, key="Thymoma_LMR")
        SII = st.number_input("SII", min_value=0.00000001, value=2.23880597
, key="Thymoma_SII")
    
    # 建立 dict（易於維護）
    input_dict = {
    "Gender": Gender,
    "Age at onset (year)": Age,
    "BMI": BMI,
    "Recurrent thymoma": Recurrent_thymoma,
    "Invasive thymoma": Invasive_thymoma,
    "Infection at admission": Infection,
    "Thyroid disease": Thyroid,
    "Autoimmune disease": Auto, 
    "Diabetes": Diabetes,
    "Hypertension": Hypertension,
    "ASCVD": ASCVD,
    "Chronic lung disease": Chronic,
    "Good syndrome": Good,
    "Disease duration (month)": Disease_duration,
    "Prednisolone daily dose before admission": Prednisolone,
    "Immunosuppressant at admission": Immunosuppressant,
    "NLR": NLR,
    "PLR": PLR,
    "LMR": LMR,
    "SII": SII
}

    if st.sidebar.button("Analysis"):
        # 用 input_dict 建立 DataFrame
       # 建立 DataFrame（按照 x_train 的欄位順序）
        input_df = pd.DataFrame([[input_dict[col] for col in x_train.columns]], columns=x_train.columns)
        # 印出模型實際特徵
        model_feature_names = model.get_booster().feature_names
        

        # 僅保留模型實際特徵
        input_df = input_df[model_feature_names]
        
        predict_and_explain(model, x_train, input_df, "Model C")

def run_model_d_page():
    st.title("Intensive care risk prediction result")
    st.markdown("""We 
provide detailed guidance through 
step-by-step instructions. Users can 
download the file below:""")
    with open("Test.pdf", "rb") as f:
        st.download_button(
            label="📥 Download of user-guide",
            data=f,
            file_name="Test.pdf",
            mime="application/pdf"
        )
    # 模型 & 資料（你之後替換正確路徑）
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(r"MG_ICU_SHAP_XGB_NonThymoma.json")
    x = pd.read_csv(r"MG_ICU_SHAP_Model_Data_SubGroup1_Thymoma_No.csv")
    x_train = x.drop(columns=[ "Y"])
    # ➤ Clinical variables
    with st.sidebar.expander("Clinical variables", expanded=True):
        Age= st.number_input("Age at onset (year)", min_value=0.00000001, value=26.0, key="NonThymoma_Age")
        Gender = st.radio(
        "Gender",
        options=[(1, "Male"), (2, "Female")],
        format_func=lambda x: x[1],index=0
        )
        Gender = Gender[0]
        Disease_duration= st.number_input("Disease duration (month)", min_value=0.00000001, value=38.0, key="NonThymoma_Disease_duration")
        BMI = st.number_input("BMI", min_value=0.00000001, value=24.96, key="NonThymoma_BMI")
    #MGFA
    # ➤ Corticosteroid variables
    with st.sidebar.expander("Treatment related variables", expanded=False):
        Prednisolone = st.number_input("Prednisolone daily dose before admission (mg)", min_value=0.0, value=30.0, key="NonThymoma_Prednisolone")
        Immunosuppressant = st.radio(
        "Immunosuppressant at admission", 
        options=[(1, "Azathioprine"), (2, "Calcineurin"), (3, "Mycophenolate"), (4, "Quinine"),(0, "None of above")], 
        format_func=lambda x: x[1],
        key="EOMG_Immuno",index=0
        )
        Immunosuppressant = Immunosuppressant[0]
    # ➤ Thymic pathology
    with st.sidebar.expander("Thymic pathology variables", expanded=False):

        Recurrent_thymoma = binary_radio("Recurrent thymoma", key="NonThymoma_Recurrent_thymoma",index = 1)
        Invasive_thymoma = binary_radio("Invasive thymoma", key="NonThymoma_Invasive_thymoma",index = 1)
    # ➤ Comorbidity variables
    with st.sidebar.expander("Comorbidity variables", expanded=False):
    
        Infection = binary_radio("Infection at admission", key="NonThymoma_Infection",index = 0)
        Thyroid = binary_radio("Thyroid disease", key="NonThymoma_Thyroid",index = 1)
        Diabetes = binary_radio("Diabetes", key="NonThymoma_Diabetes",index = 1)
        Hypertension = binary_radio("Hypertension", key="NonThymoma_Hypertension",index = 1)
        Auto = binary_radio("Autoimmune disease", key="NonThymoma_Auto",index = 1)
        ASCVD = binary_radio("ASCVD", key="NonThymoma_ASCVD",index = 1)
        Chronic = binary_radio("Chronic lung disease", key="NonThymoma_Chronic",index = 1)
        Good = binary_radio("Good syndrome", key="NonThymoma_Good",index = 1)

    # ➤ Systemic inflammation markers profile
    with st.sidebar.expander("Systemic inflammation markers profile variables", expanded=False):
        #WBC
        NLR = st.number_input("NLR", min_value=0.00000001, value=10.75
, key="NonThymoma_NLR")
        PLR = st.number_input("PLR", min_value=0.00000001, value=455.3571429
, key="NonThymoma_PLR")
        LMR = st.number_input("LMR", min_value=0.00000001, value=2.47008547
, key="NonThymoma_LMR")
        SII = st.number_input("SII", min_value=0.00000001, value=3289500.0
, key="NonThymoma_SII")
    
    # 建立 dict（易於維護）
    input_dict = {
    "Gender": Gender,
    "Age at onset (year)": Age,
    "BMI": BMI,
    "Recurrent thymoma": Recurrent_thymoma,
    "Invasive thymoma": Invasive_thymoma,
    "Infection at admission": Infection,
    "Thyroid disease": Thyroid,
    "Autoimmune disease": Auto, 
    "Diabetes": Diabetes,
    "Hypertension": Hypertension,
    "ASCVD": ASCVD,
    "Chronic lung disease": Chronic,
    "Good syndrome": Good,
    "Disease duration (month)": Disease_duration,
    "MGFA clinical classification": 0,
    "Prednisolone daily dose before admission": Prednisolone,
    "Immunosuppressant at admission": Immunosuppressant,
    "NLR": NLR,
    "PLR": PLR,
    "LMR": LMR,
    "SII": SII
}

    if st.sidebar.button("Analysis"):
        # 用 input_dict 建立 DataFrame
       # 建立 DataFrame（按照 x_train 的欄位順序）
        input_df = pd.DataFrame([[input_dict[col] for col in x_train.columns]], columns=x_train.columns)
        # 印出模型實際特徵
        model_feature_names = model.get_booster().feature_names
        

        # 僅保留模型實際特徵
        input_df = input_df[model_feature_names]
        
        predict_and_explain(model, x_train, input_df, "Model D")
# ------------------------- 主控制邏輯 -------------------------

if model_choice == "EOMG":
    run_model_a_page()
elif model_choice == "LOMG":
    run_model_b_page()
elif model_choice == "Thymoma":
    run_model_c_page()
elif model_choice == "Non-Thymoma":
    run_model_d_page()


   
