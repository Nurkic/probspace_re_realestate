import pandas as pd
import category_encoders as ce

import json

def rename_t(df) -> pd.DataFrame:
    with open("../input/names.json", "r", encoding="utf-8") as f:
        d = json.load(f)
    df = df.rename(columns=d)

    return df

def new_df(
    df1: pd.DataFrame,
    df2: pd.DataFrame
) -> pd.DataFrame:
    df1 = df1.copy()
    df2 = df2.copy()
    df2 = rename_t(df2)
    """target_columns = ['地域', '種類', '市区町村コード', '面積（㎡）','建ぺい率（％）', '容積率（％）',
                    '最寄駅：距離（分）', '前面道路：方位', '地区詳細','土地の形状','市区町村名',
                    '前面道路：種類', '前面道路：幅員（ｍ）',  '間口', '建物の構造', '最寄駅：名称',
                    '用途', '都市計画', '建築年','y']"""
    target_columns = ['Type','Region','MunicipalityCode','Area','CoverageRatio','FloorAreaRatio',
                    'TimeToNearestStation','Direction', 'DistrictDetails','LandShape','Municipality',
                    'Classification','Breadth','Frontage','Structure','NearestStation',
                    'Use','CityPlanning', 'BuildingYear', 'y']
    #print(df2.columns)
    tika = df2[target_columns]
    tika = rename_t(df2)

    #df1 = df1[target_columns+['延床面積（㎡）', '改装', '間取り', '取引の事情等']]
    df1 = df1[target_columns + ["TotalFloorArea", "Renovation", "FloorPlan", "Remarks"]]
    df1 = df1.reset_index(drop=True)

    #df1['間取り数'] = pd.to_numeric(df1['間取り'].str[0], errors='raise')
    #df1['NumberOfRooms'] = df1['間取り'].str[2:].str.len()//2+1+df1['間取り数']

    #cols = ['地域', '市区町村名', '地区詳細', '建ぺい率（％）', '容積率（％）', '都市計画', '前面道路：方位', '前面道路：種類', '最寄駅：名称']
    cols = ['Region', 'Municipality', 'DistrictDetails', 'CoverageRatio', 'CityPlanning', 'Direction', 'Classification', 'NearestStation']
        
    ce_oe = ce.OrdinalEncoder(cols=cols,handle_unknown='impute')
    tika = ce_oe.fit_transform(tika)

    """for obj_col in object_cols:
        df[obj_col] = df[obj_col].astype("category")"""

    for col in cols:
        #print(col, df1.shape)
        #print(df1[col].head(), tika[col].head())
        df1 = pd.merge(df1, tika[[col, 'y']].groupby(col).mean().rename(columns={'y':col+'_y'}), on=col, how='left')
    #print(df1.columns)
    all_df_target = df1
    all_df_target.loc[all_df_target['DistrictDetails_y'].isna(), 'DistrictDetails_y'] = all_df_target.loc[all_df_target['DistrictDetails_y'].isna(), 'Municipality_y']
    all_df_target.loc[all_df_target['NearestStation_y'].isna(), 'NearestStation_y'] = all_df_target.loc[all_df_target['NearestStation_y'].isna(), 'DistrictDetails_y']
    
    cols = ['Municipality', 'DistrictDetails', 'NearestStation']
    #cols = ['地域', '市区町村コード', '地区詳細', '建ぺい率（％）', '容積率（％）', '都市計画', '前面道路：方位', '前面道路：種類', '最寄駅：名称']
    for col in cols:
        all_df_target['m2x'+col] = all_df_target[col+'_y'] * all_df_target['Area']/100
        #all_df_target['nm2x'+col] = all_df_target[col+'y'] * all_df_target['延床面積（㎡）']/100
        all_df_target['m2m2x'+col] = all_df_target[col+'_y'] * (all_df_target['Area']+all_df_target['TotalFloorArea'].fillna(0))/100
        all_df_target['m2m2x'+col+'_sta'] = all_df_target['m2m2x'+col] * (1 - all_df_target['TimeToNearestStation'].clip(0, 10)*0.02)
        all_df_target = all_df_target.loc[:,~all_df_target.columns.duplicated()]
    return all_df_target
