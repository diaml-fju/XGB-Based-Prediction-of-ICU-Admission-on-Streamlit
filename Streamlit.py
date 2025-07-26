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
def binary_radio(label):
    return st.sidebar.radio(
        label,
        options=[0, 1],
        format_func=lambda x: f"{x} (Yes)" if x == 1 else f"{x} (No)"
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
    Age = st.sidebar.number_input("Age at onset (year)", 50,disabled=True)
    Gender = st.sidebar.radio(
    "Gender",
    options=[(1, "1 (Male)"), (2, "2 (Female)")],
    format_func=lambda x: x[1]
    )
    Gender = Gender[0]  
    BMI = st.sidebar.number_input("BMI", min_value=0.00000001, value=1.0)

    Infection = binary_radio("Infection at admission")
    Thyroid = binary_radio("Thyroid disease")
    Auto = binary_radio("Autoimmune disease")
    Diabetes = binary_radio("Diabetes")
    Hypertension = binary_radio("Hypertension")
    ASCVD = binary_radio("ASCVD")
    Chronic = binary_radio("Chronic lung disease")
    Good = binary_radio("Good syndrome")

    Disease_duration= st.sidebar.number_input("Disease duration (month)", min_value=0.00000001, value=1.0)
    Prednisolone = st.sidebar.number_input("Prednisolone daily dose before admission (mg)", min_value=0.00000001, value=1.0)
    #Immunosuppressant = st.sidebar.radio("Immunosuppressant at admission", 0, 3, 0)
    Immunosuppressant = st.sidebar.radio(
    "Immunosuppressant at admission", 
    options=[0, 1, 2, 3, 4], 
    index=0
)
    Anti_MuSK = binary_radio("Anti-MuSK")
    Anti_AChR = binary_radio("Anti-AChR")
    dSN = binary_radio("dSN")
    Thymoma = binary_radio("Thymoma")
    Thymic = binary_radio("Thymic hyperplasia")
    Thymectomy = binary_radio("Thymectomy")
    NLR = st.sidebar.number_input("NLR", min_value=0.00000001, value=1.0)
    PLR = st.sidebar.number_input("PLR", min_value=0.00000001, value=1.0)
    LMR = st.sidebar.number_input("LMR", min_value=0.00000001, value=1.0)
    SII = st.sidebar.number_input("SII", min_value=0.00000001, value=1.0)

    
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
    # 輸入變數
    Age = st.sidebar.number_input("Age at onset (year)", 0, 50,disabled=True)
    Gender = st.sidebar.radio(
    "Gender",
    options=[(1, "1 (Male)"), (2, "2 (Female)")],
    format_func=lambda x: x[1]
    )
    Gender = Gender[0]
    BMI = st.sidebar.number_input("BMI" ,min_value=0.00000001, value=1.0)

    Infection = binary_radio("Infection at admission")
    Thyroid = binary_radio("Thyroid disease")
    Auto = binary_radio("Autoimmune disease")
    Diabetes = binary_radio("Diabetes")
    Hypertension = binary_radio("Hypertension")
    ASCVD = binary_radio("ASCVD")
    Chronic = binary_radio("Chronic lung disease")
    Good = binary_radio("Good syndrome")

    Disease_duration= st.sidebar.number_input("Disease duration (month)", min_value=0.00000001, value=1.0)
    Prednisolone = st.sidebar.number_input("Prednisolone daily dose before admission (mg)", min_value=0.00000001, value=1.0)
    Immunosuppressant = st.sidebar.number_input("Immunosuppressant at admission", min_value=0.00000001, value=1.0)

    Anti_MuSK = binary_radio("Anti-MuSK")
    Anti_AChR = binary_radio("Anti-AChR")
    dSN = binary_radio("dSN")

    Thymoma = st.sidebar.number_input("Thymoma", min_value=0.00000001, value=1.0)

    Thymic = binary_radio("Thymic hyperplasia")
    Thymectomy = st.sidebar.number_input("Thymectomy", min_value=0.00000001, value=1.0)

    NLR = st.sidebar.number_input("NLR", min_value=0.00000001, value=1.0)
    PLR = st.sidebar.number_input("PLR", min_value=0.00000001, value=1.0)
    LMR = st.sidebar.number_input("LMR", min_value=0.00000001, value=1.0)
    SII = st.sidebar.number_input("SII", min_value=0.00000001, value=1.0)

    
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
    st.sidebar.markdown("### Clinical variables")
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
    Immunosuppressant = st.sidebar.number_input("Immunosuppressant at admission", min_value=0.00000001, value=1.0)
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


   
