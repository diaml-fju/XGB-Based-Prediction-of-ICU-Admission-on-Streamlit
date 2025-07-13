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
    Gender = st.sidebar.radio("Gender", options=[1, 2])
    BMI = st.sidebar.number_input("BMI", 10.0, 50.0, 22.5)
    Infection = st.sidebar.radio("Infection at admission", options=[0, 1])  # 0 = No, 1 = Yes
    Thyroid = st.sidebar.radio("Thyroid disease", options=[0, 1])  # 0 = No, 1 = Yes
    Auto = st.sidebar.radio("Autoimmune disease", options=[0, 1])  # 0 = No, 1 = Yes
    Diabetes = st.sidebar.radio("Diabetes", options=[0, 1])  # 0 = No, 1 = Yes
    Hypertension = st.sidebar.radio("Hypertension", options=[0, 1])  # 0 = No, 1 = Yes
    ASCVD = st.sidebar.radio("ASCVD", options=[0, 1])  # 0 = No, 1 = Yes
    Chronic = st.sidebar.radio("Chronic lung disease", options=[0, 1])  # 0 = No, 1 = Yes
    Good = st.sidebar.radio("Good syndrome", options=[0, 1])  # 0 = No, 1 = Yes
    Disease_duration= st.sidebar.number_input("Disease duration (month)", 0, 120, 0)

    Prednisolone = st.sidebar.number_input("Prednisolone daily dose before admission", 0, 100, 0)
    Immunosuppressant = st.sidebar.number_input("Immunosuppressant at admission", 0, 3, 0)
    Anti_MuSK = st.sidebar.radio("Anti-MuSK", options=[0, 1])  # 0 = No, 1 = Yes
    Anti_AChR = st.sidebar.radio("Anti-AChR", options=[0, 1])  # 0 = No, 1 = Yes
    dSN = st.sidebar.radio("dSN", options=[0, 1])  # 0 = No, 1 = Yes
    Thymoma = st.sidebar.number_input("Thymoma", 0, 4, 0)
    Thymic = st.sidebar.radio("Thymic hyperplasia", options=[0, 1])  # 0 = No, 1 = Yes
    Thymectomy = st.sidebar.number_input("Thymectomy", 0, 3, 0)

    NLR = st.sidebar.number_input("NLR", 0.0, 100.0, 0.0)
    PLR = st.sidebar.number_input("PLR", 0.0, 1000.0, 0.0)
    LMR = st.sidebar.number_input("LMR", 0.0, 20.0, 0.0)
    SII = st.sidebar.number_input("SII", 0.0, 10000000.0, 0.0)
    
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
    # 輸入變數
    Gender = st.sidebar.radio("Gender", options=[1, 2])
    BMI = st.sidebar.number_input("BMI", 10.0, 50.0, 22.5)
    Infection = st.sidebar.radio("Infection at admission", options=[0, 1])  # 0 = No, 1 = Yes
    Thyroid = st.sidebar.radio("Thyroid disease", options=[0, 1])  # 0 = No, 1 = Yes
    Auto = st.sidebar.radio("Autoimmune disease", options=[0, 1])  # 0 = No, 1 = Yes
    Diabetes = st.sidebar.radio("Diabetes", options=[0, 1])  # 0 = No, 1 = Yes
    Hypertension = st.sidebar.radio("Hypertension", options=[0, 1])  # 0 = No, 1 = Yes
    ASCVD = st.sidebar.radio("ASCVD", options=[0, 1])  # 0 = No, 1 = Yes
    Chronic = st.sidebar.radio("Chronic lung disease", options=[0, 1])  # 0 = No, 1 = Yes
    Good = st.sidebar.radio("Good syndrome", options=[0, 1])  # 0 = No, 1 = Yes
    Disease_duration= st.sidebar.number_input("Disease duration (month)", 0, 120, 0)

    Prednisolone = st.sidebar.number_input("Prednisolone daily dose before admission", 0, 100, 0)
    Immunosuppressant = st.sidebar.number_input("Immunosuppressant at admission", 0, 3, 0)
    Anti_MuSK = st.sidebar.radio("Anti-MuSK", options=[0, 1])  # 0 = No, 1 = Yes
    Anti_AChR = st.sidebar.radio("Anti-AChR", options=[0, 1])  # 0 = No, 1 = Yes
    dSN = st.sidebar.radio("dSN", options=[0, 1])  # 0 = No, 1 = Yes
    Thymoma = st.sidebar.number_input("Thymoma", 0, 4, 0)
    Thymic = st.sidebar.radio("Thymic hyperplasia", options=[0, 1])  # 0 = No, 1 = Yes
    Thymectomy = st.sidebar.number_input("Thymectomy", 0, 3, 0)

    NLR = st.sidebar.number_input("NLR", 0.0, 100.0, 0.0)
    PLR = st.sidebar.number_input("PLR", 0.0, 1000.0, 0.0)
    LMR = st.sidebar.number_input("LMR", 0.0, 20.0, 0.0)
    SII = st.sidebar.number_input("SII", 0.0, 10000000.0, 0.0)
    
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
    Gender = st.sidebar.radio("Gender", options=[1, 2])
    Age= st.sidebar.number_input("Age at onset (year)", 0, 100, 1)
    BMI = st.sidebar.number_input("BMI", 10.0, 50.0, 22.5)
    Recurrent_thymoma = st.sidebar.radio("Recurrent thymoma", options=[0, 1])  # 0 = No, 1 = Yes
    Invasive_thymoma = st.sidebar.radio("Invasive thymoma", options=[0, 1])  # 0 = No, 1 = Yes
    Infection = st.sidebar.radio("Infection at admission", options=[0, 1])  # 0 = No, 1 = Yes
    Thyroid = st.sidebar.radio("Thyroid disease", options=[0, 1])  # 0 = No, 1 = Yes
    Auto = st.sidebar.radio("Autoimmune disease", options=[0, 1])  # 0 = No, 1 = Yes
    Diabetes = st.sidebar.radio("Diabetes", options=[0, 1])  # 0 = No, 1 = Yes
    Hypertension = st.sidebar.radio("Hypertension", options=[0, 1])  # 0 = No, 1 = Yes
    ASCVD = st.sidebar.radio("ASCVD", options=[0, 1])  # 0 = No, 1 = Yes
    Chronic = st.sidebar.radio("Chronic lung disease", options=[0, 1])  # 0 = No, 1 = Yes
    Good = st.sidebar.radio("Good syndrome", options=[0, 1])  # 0 = No, 1 = Yes
    Disease_duration= st.sidebar.number_input("Disease duration (month)", 0, 300, 0)
    Prednisolone = st.sidebar.number_input("Prednisolone daily dose before admission", 0, 100, 0)
    Immunosuppressant = st.sidebar.number_input("Immunosuppressant at admission", 0, 4, 0)
    NLR = st.sidebar.number_input("NLR", 0.0, 100.0, 0.0)
    PLR = st.sidebar.number_input("PLR", 0.0, 2000.0, 0.0)
    LMR = st.sidebar.number_input("LMR", 0.0, 40.0, 0.0)
    SII = st.sidebar.number_input("SII", 0.0, 10000000.0, 0.0)
    
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
    Gender = st.sidebar.radio("Gender", options=[1, 2])
    Age= st.sidebar.number_input("Age at onset (year)", 0, 100, 1)
    BMI = st.sidebar.number_input("BMI", 10.0, 50.0, 22.5)
    Recurrent_thymoma = st.sidebar.radio("Recurrent thymoma", options=[0, 1])  # 0 = No, 1 = Yes
    Invasive_thymoma = st.sidebar.radio("Invasive thymoma", options=[0, 1])  # 0 = No, 1 = Yes
    Infection = st.sidebar.radio("Infection at admission", options=[0, 1])  # 0 = No, 1 = Yes
    Thyroid = st.sidebar.radio("Thyroid disease", options=[0, 1])  # 0 = No, 1 = Yes
    Auto = st.sidebar.radio("Autoimmune disease", options=[0, 1])  # 0 = No, 1 = Yes
    Diabetes = st.sidebar.radio("Diabetes", options=[0, 1])  # 0 = No, 1 = Yes
    Hypertension = st.sidebar.radio("Hypertension", options=[0, 1])  # 0 = No, 1 = Yes
    ASCVD = st.sidebar.radio("ASCVD", options=[0, 1])  # 0 = No, 1 = Yes
    Chronic = st.sidebar.radio("Chronic lung disease", options=[0, 1])  # 0 = No, 1 = Yes
    Good = st.sidebar.radio("Good syndrome", options=[0, 1])  # 0 = No, 1 = Yes
    Disease_duration= st.sidebar.number_input("Disease duration (month)", 0, 300, 0)
    Prednisolone = st.sidebar.number_input("Prednisolone daily dose before admission", 0, 100, 0)
    Immunosuppressant = st.sidebar.number_input("Immunosuppressant at admission", 0, 4, 0)
    NLR = st.sidebar.number_input("NLR", 0.0, 100.0, 0.0)
    PLR = st.sidebar.number_input("PLR", 0.0, 2000.0, 0.0)
    LMR = st.sidebar.number_input("LMR", 0.0, 40.0, 0.0)
    SII = st.sidebar.number_input("SII", 0.0, 10000000.0, 0.0)
    
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


   
