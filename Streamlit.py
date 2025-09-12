import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap

# ------------------------- 分頁切換 -------------------------
model_choice = st.sidebar.selectbox("Model", [
    "EOMG",
    "LOMG",
    "Thymoma",
    "Non-Thymoma"
])
#tab1, tab2, tab3, tab4 = st.tabs(["EOMG", "LOMG", "Thymoma", "Non-Thymoma"])


# ------------------------- 共用函數：預測 + SHAP -------------------------
def predict_and_explain(model, x_train, input_df, model_name):
    import shap
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import streamlit as st
    import xgboost as xgb
    st.subheader("Predict result")

    try:
        # 特徵對齊
        model_feature_names = model.get_booster().feature_names
        input_df = input_df[model_feature_names]
        background = x_train[model_feature_names]

        # 預測
        proba = model.predict_proba(input_df)[0]
        pred_class = int(np.argmax(proba))
        

        if pred_class == 1:
            st.error("Predict result：ICU admission")
        else:
            st.success("Predict result：Not ICU admission")

        # SHAP 解釋
        explainer = shap.TreeExplainer(model, data=background,model_output="probability", feature_perturbation="interventional")
        shap_values = explainer.shap_values(input_df)
        # ✅ 防止 index 錯誤
        shap_val = shap_values[0]
        #st.write("Shap_values",shap_values)
        #st.write("SHAP",shap_val)
        base_val = explainer.expected_value
        st.subheader("SHAP Waterfall explanation")
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
def binary_radio(label,key= None):
    return st.radio(
        label,
        options=[0, 1],
        format_func=lambda x: f"Yes" if x == 1 else f"No",
        key=key
    )




# ------------------------- 模型 A -------------------------
def run_model_a_page():
    st.title("Model EOMG prediction page")
    # 模型 & 資料（你之後替換正確路徑）
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(r"MG_ICU_SHAP_XGB_EOMG.json")
    x = pd.read_csv(r"MG_ICU_SHAP_Model_Data_SubGroup2_Age50D_New_FeaName.csv")
    x_train = x.drop(columns=[ "Y","MGFA clinical classification"])
    # 輸入變數
    # ➤ Clinical variables
    with st.sidebar.expander("Clinical variables", expanded=True):
        Age = st.number_input("Age at onset (year)", 50,disabled=True)
        Gender = st.radio(
        "Gender",
        options=[(1, "Male"), (2, "Female")],
        format_func=lambda x: x[1],
        key="EOMG_Gender"
        )
        Gender = Gender[0]  
        Disease_duration= st.number_input("Disease duration (month)", min_value=0.01, value=1.0, key="EOMG_Disease_duration")
        BMI = st.number_input("BMI", min_value=0.01, value=1.0)

    #MGFA
    # ➤ Corticosteroid variables
    with st.sidebar.expander("Corticosteroid variables", expanded=False):
    
        Prednisolone = st.number_input("Prednisolone daily dose before admission (mg)", min_value=0.01, value=1.0)
        Immunosuppressant = st.radio(
        "Immunosuppressant at admission", 
        options=[0, 1, 2, 3, 4], 
        index=0,
        key="EOMG_Immuno"
        )
    # ➤ Thymic pathology
    with st.sidebar.expander("Thymic pathology variables", expanded=False):
    
        Thymoma = binary_radio("Thymoma", key="EOMG_Thymoma")
        Thymic = binary_radio("Thymic hyperplasia", key="EOMG_Thymic")
        Thymectomy = binary_radio("Thymectomy", key="EOMG_Thymectomy")
    
    # ➤ Serology
    with st.sidebar.expander("Serology of autoantibody", expanded=False):

        Anti_AChR = binary_radio("Anti-AChR", key="EOMG_Anti_AChR")
        Anti_MuSK = binary_radio("Anti-MuSK", key="EOMG_Anti_MuSK")
        dSN = binary_radio("dSN")

    # ➤ Comorbidity
    with st.sidebar.expander("Comorbidity variables", expanded=False):
        Infection = binary_radio("Infection at admission", key="EOMG_Infection")
        Thyroid = binary_radio("Thyroid disease", key="EOMG_Thyroid")
        Diabetes = binary_radio("Diabetes", key="EOMG_Diabetes")
        Hypertension = binary_radio("Hypertension", key="EOMG_Hypertension")
        Auto = binary_radio("Autoimmune disease", key="EOMG_Auto")
        ASCVD = binary_radio("ASCVD", key="EOMG_ASCVD")
        Chronic = binary_radio("Chronic lung disease", key="EOMG_Chronic")
        Good = binary_radio("Good syndrome", key="EOMG_Good")

    # ➤ Inflammation
    with st.sidebar.expander("Systemic inflammation markers", expanded=False):
        NLR = st.number_input("NLR", min_value=0.01, value=1.0, key="EOMG_NLR")
        PLR = st.number_input("PLR", min_value=0.01, value=1.0, key="EOMG_PLR")
        LMR = st.number_input("LMR", min_value=0.01, value=1.0, key="EOMG_LMR")
        SII = st.number_input("SII", min_value=0.01, value=1.0, key="EOMG_SII")
    
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

    


    if st.sidebar.button("Predict"):
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
    st.title("Model LOMG prediction page")
    # 模型 & 資料（你之後替換正確路徑）
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(r"MG_ICU_SHAP_XGB_LOMG.json")
    x = pd.read_csv(r"MG_ICU_SHAP_Model_Data_SubGroup2_Age50U_New_FeaName.csv")
    x_train = x.drop(columns=[ "Y","MGFA clinical classification"])
    # ➤ Clinical variables
    with st.sidebar.expander("Clinical variables", expanded=True):
        Age = st.number_input("Age at onset (year)", 50, disabled=True)
        Gender = st.radio(
            "Gender",
            options=[(1, "1 (Male)"), (2, "2 (Female)")],
            format_func=lambda x: x[1]
        )
        Gender = Gender[0]
        Disease_duration = st.number_input("Disease duration (month)", min_value=0.01, value=1.0)
        BMI = st.number_input("BMI", min_value=0.01, value=1.0)

    # ➤ Corticosteroid variables
    with st.sidebar.expander("Corticosteroid variables", expanded=False):
        Prednisolone = st.number_input("Prednisolone daily dose before admission (mg)", min_value=0.01, value=1.0)
        Immunosuppressant = st.radio("Immunosuppressant at admission", options=[0, 1, 2, 3, 4], index=0)

    # ➤ Thymic pathology
    with st.sidebar.expander("Thymic pathology variables", expanded=False):
        Thymoma = binary_radio("Thymoma")
        Thymic = binary_radio("Thymic hyperplasia")
        Thymectomy = binary_radio("Thymectomy")

    # ➤ Serology
    with st.sidebar.expander("Serology of autoantibody", expanded=False):
        Anti_AChR = binary_radio("Anti-AChR")
        Anti_MuSK = binary_radio("Anti-MuSK")
        dSN = binary_radio("dSN")

    # ➤ Comorbidity
    with st.sidebar.expander("Comorbidity variables", expanded=False):
        Infection = binary_radio("Infection at admission")
        Thyroid = binary_radio("Thyroid disease")
        Diabetes = binary_radio("Diabetes")
        Hypertension = binary_radio("Hypertension")
        Auto = binary_radio("Autoimmune disease")
        ASCVD = binary_radio("ASCVD")
        Chronic = binary_radio("Chronic lung disease")
        Good = binary_radio("Good syndrome")

    # ➤ Inflammation
    with st.sidebar.expander("Systemic inflammation markers", expanded=False):
        NLR = st.number_input("NLR", min_value=0.01, value=1.0)
        PLR = st.number_input("PLR", min_value=0.01, value=1.0)
        LMR = st.number_input("LMR", min_value=0.01, value=1.0)
        SII = st.number_input("SII", min_value=0.01, value=1.0)

    
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

    if st.sidebar.button("Predict"):
        # 用 input_dict 建立 DataFrame
       # 建立 DataFrame（按照 x_train 的欄位順序）
        input_df = pd.DataFrame([[input_dict[col] for col in x_train.columns]], columns=x_train.columns)
        # 印出模型實際特徵
        model_feature_names = model.get_booster().feature_names
        

        # 僅保留模型實際特徵
        input_df = input_df[model_feature_names]
        
        predict_and_explain(model, x_train, input_df, "Model B")

def run_model_c_page():
    st.title("Model Thymoma prediction page")
    # 模型 & 資料（你之後替換正確路徑）
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(r"MG_ICU_SHAP_XGB_Thymoma.json")
    x = pd.read_csv(r"MG_ICU_SHAP_Model_Data_SubGroup1_X9_1_FeaName.csv")
    x_train = x.drop(columns=[ "Y"])
    # 輸入變數
    st.sidebar.markdown('---')
    st.sidebar.markdown("### Clinical variables")
    Age= st.sidebar.number_input("Age at onset (year)", min_value=0.00000001, value=1.0)
    Gender = st.sidebar.radio(
    "Gender",
    options=[(1, "1 (Male)"), (2, "2 (Female)")],
    format_func=lambda x: x[1]
    )
    Gender = Gender[0]
    Disease_duration= st.sidebar.number_input("Disease duration (month)", min_value=0.00000001, value=1.0)
    BMI = st.sidebar.number_input("BMI", min_value=0.00000001, value=1.0)
    #MGFA

    st.sidebar.markdown('---')
    st.sidebar.markdown("### Corticosteroid variables")
    Prednisolone = st.sidebar.number_input("Prednisolone daily dose before admission", min_value=0.00000001, value=1.0)
    Immunosuppressant = st.sidebar.number_input("Immunosuppressant at admission", min_value=0.00000001, value=1.0)

    st.sidebar.markdown('---')
    st.sidebar.markdown("### Thymic pathology variables")

    Recurrent_thymoma = binary_radio("Recurrent thymoma")
    Invasive_thymoma = binary_radio("Invasive thymoma")

    st.sidebar.markdown('---')
    st.sidebar.markdown("### Comorbidity variables")
    Infection = binary_radio("Infection at admission")
    Thyroid = binary_radio("Thyroid disease")
    Diabetes = binary_radio("Diabetes")
    Hypertension = binary_radio("Hypertension")
    Auto = binary_radio("Autoimmune disease")
    ASCVD = binary_radio("ASCVD")
    Chronic = binary_radio("Chronic lung disease")
    Good = binary_radio("Good syndrome")
    
    st.sidebar.markdown('---')
    st.sidebar.markdown("### Systemic inflammation markers profile")
    #WBC
    NLR = st.sidebar.number_input("NLR", min_value=0.00000001, value=1.0)
    PLR = st.sidebar.number_input("PLR", min_value=0.00000001, value=1.0)
    LMR = st.sidebar.number_input("LMR", min_value=0.00000001, value=1.0)
    SII = st.sidebar.number_input("SII", min_value=0.00000001, value=1.0)
    
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

    if st.sidebar.button("Predict"):
        # 用 input_dict 建立 DataFrame
       # 建立 DataFrame（按照 x_train 的欄位順序）
        input_df = pd.DataFrame([[input_dict[col] for col in x_train.columns]], columns=x_train.columns)
        # 印出模型實際特徵
        model_feature_names = model.get_booster().feature_names
        

        # 僅保留模型實際特徵
        input_df = input_df[model_feature_names]
        
        predict_and_explain(model, x_train, input_df, "Model C")

def run_model_d_page():
    st.title("Model NonThymoma prediction page")
    # 模型 & 資料（你之後替換正確路徑）
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(r"MG_ICU_SHAP_XGB_NonThymoma.json")
    x = pd.read_csv(r"MG_ICU_SHAP_Model_Data_SubGroup1_X9_0_FeaName.csv")
    x_train = x.drop(columns=[ "Y"])
    # 輸入變數
    Gender = st.sidebar.radio(
    "Gender",
    options=[(1, "1 (Male)"), (2, "2 (Female)")],
    format_func=lambda x: x[1]
    )
    Gender = Gender[0]
    Age= st.sidebar.number_input("Age at onset (year)", min_value=0.00000001, value=1.0)
    BMI = st.sidebar.number_input("BMI", min_value=0.00000001, value=1.0)
    Recurrent_thymoma = binary_radio("Recurrent thymoma")
    Invasive_thymoma = binary_radio("Invasive thymoma")
    Infection = binary_radio("Infection at admission")
    Thyroid = binary_radio("Thyroid disease")
    Auto = binary_radio("Autoimmune disease")
    Diabetes = binary_radio("Diabetes")
    Hypertension = binary_radio("Hypertension")
    ASCVD = binary_radio("ASCVD")
    Chronic = binary_radio("Chronic lung disease")
    Good = binary_radio("Good syndrome")
    Disease_duration= st.sidebar.number_input("Disease duration (month)", min_value=0.00000001, value=1.0)
    Prednisolone = st.sidebar.number_input("Prednisolone daily dose before admission", min_value=0.00000001, value=1.0)
    Immunosuppressant = st.sidebar.number_input("Immunosuppressant at admission", 0, 4, 0)
    NLR = st.sidebar.number_input("NLR", min_value=0.00000001, value=1.0)
    PLR = st.sidebar.number_input("PLR", min_value=0.00000001, value=1.0)
    LMR = st.sidebar.number_input("LMR", min_value=0.00000001, value=1.0)
    SII = st.sidebar.number_input("SII", min_value=0.00000001, value=1.0)
    
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

    if st.sidebar.button("Predict"):
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


   
