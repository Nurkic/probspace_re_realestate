import numpy as np
import pandas as pd

import json
import re

import category_encoders as ce

class Prep_publish:
    def __init__(
        self,
        df: pd.DataFrame
    ) -> None:
        self.df = df

    """ Unify published column names"""
    def rename(self) -> pd.DataFrame:
        df = self.df.copy()
        pair = [("所在地コード","市区町村コード"),("建蔽率","建ぺい率（％）"),("容積率","容積率（％）"),("駅名","最寄駅：名称"), 
        ("地積","面積（㎡）"),("市区町村名","市区町村名"),('前面道路の幅員','前面道路：幅員（ｍ）'), 
        ("前面道路の方位区分","前面道路：方位"),("前面道路区分","前面道路：種類"),("形状区分","土地の形状"), 
        ("用途区分","都市計画"), ('用途','地域')]
        pair_dict = {key:value for key, value in pair}

        df = df.rename(columns=pair_dict)
        return df

    def area(self) -> pd.DataFrame:
        df = self.df.copy()
        df["面積（㎡）"] = df["面積（㎡）"].clip(0, 3000)

        return df

    def create_features(self) -> pd.DataFrame:
        df = self.df.copy()
        ['その他',  '住宅', '店舗', '事務所', '作業場', '倉庫', '共同住宅', '工場', '駐車場']
        [         '住宅', '店舗', '事務所', '銀行', '旅館', '給油所', '工場', '倉庫', '農地', '山林', '医院', '空地', '作業場', '原野', 'その他', '用材', '雑木']
        riyo_list = np.array(['住宅', '店舗', '事務所', '_', '_', '_', '工場', '倉庫', '_', '_', '_', '_', '作業場', '_', 'その他', '_', '_'])
        print(riyo_list)
        print(df["利用の現況"].head())
        riyo_now = [[0]*(17 - len(str(num)))+list(map(int, list(str(num)))) for num in df['利用の現況'].values]
        riyo_now = np.array(riyo_now)
        print(riyo_now)
        print(len(riyo_now))
        riyo_lists = ['、'.join(riyo_list[onehot.astype('bool')]) for onehot in riyo_now]
        for i in range(len(riyo_lists)):
            if 'その他' in riyo_lists[i]:
                riyo_lists[i] = riyo_lists[i].replace('その他', df.loc[i, '利用状況表示'])
            riyo_lists[i] = riyo_lists[i].replace('_', 'その他').replace('、雑木林', '').replace('、診療所', '').replace('、車庫', '').replace('、集会場', '')\
            .replace('、寄宿舎', '').replace('、駅舎', '').replace('、劇場', '').replace('、物置', '').replace('、集会場', '').replace('、映画館', '')\
            .replace('、遊技場', '').replace('兼', '、').replace('、建築中', 'その他').replace('、試写室', '').replace('、寮', '').replace('、保育所', '')\
            .replace('、治療院', '').replace('、診療所', '').replace('、荷捌所', '').replace('建築中', 'その他').replace('事業所', '事務所').replace('、営業所', '')
        df['利用の現況'] = riyo_lists
        print(df["利用の現況"].head())
        return df

    def create_new_df(self) -> pd.DataFrame:
        df = self.df.copy()
        kakakus = []
        for i, year in enumerate(df.columns[-37:-38:-1]):
            df_sub = pd.merge(df[df.columns[:-75:]], df[df[year]>0][['id', year]], on='id')
            df_sub['取引時点'] = 2019-i
            df_sub = df_sub.rename(columns={year: 'y'})
            df_sub['y'] = df_sub['y'] /100000
            kakakus.append(df_sub)
        published_df2 = pd.concat(kakakus)

        published_df2['最寄駅：距離（分）'] = published_df2['駅距離']//50
        published_df2['最寄駅：距離（分）'] = published_df2['最寄駅：距離（分）'].map(lambda x: 120 if x>120 else x)
        published_df2['間口（比率）'] = published_df2['間口（比率）'].clip(10, 100)
        published_df2['奥行（比率）'] = published_df2['奥行（比率）'].clip(10, 100)
        published_df2['間口'] = np.sqrt(published_df2['面積（㎡）']/published_df2['間口（比率）']/published_df2['奥行（比率）'])*published_df2['間口（比率）']

        published_df2['種類'] = np.nan
        published_df2['建築年'] = np.nan
        published_df2['建物の構造'] = published_df2['建物構造'].str.replace('SRC','ＳＲＣ').str.replace('RC','ＲＣ').str.replace('W','木造').str.replace('LS','軽量鉄骨造').str.replace('S','鉄骨造')
        published_df2['建物の構造'] = published_df2['建物の構造'].str.replace('[0-9]','').str.replace('FB','').str.replace('B','_').value_counts()
        published_df2['最寄駅：名称'] = published_df2['最寄駅：名称'].str.replace('ケ','ヶ')

        se = published_df2['住居表示'].str.strip('東京都').str.replace('大字', '').str.replace('字', '')
        for num in ['１', '２', '３', '４', '５', '６', '７', '８', '９']:
            se = se.str.split(num).str[0].str.strip()
        published_df2['DistrictDetails'] = se
        published_df2['DistrictDetails'] = published_df2['DistrictDetails'].str[:5]
        published_df2['市区町村名'] = published_df2['市区町村名'].str.strip('市区町村').str.strip('西多摩郡東京')

        rep ={'1低専':'第１種低層住居専用地域', '2低専':'第２種低層住居専用地域', '1中専':'第１種中高層住居専用地域', '2中専':'第２種中高層住居専用地域', '1住居':'第１種住居地域',
            '2住居':'第２種住居地域', '準住居':'準住居地域', '商業':'商業地域', '近商':'近隣商業地域', '工業':'工業地域', '工専':'工業専用地域',  '準工':'準工業地域', '田園住':'田園住居地域'}
        for key, value in rep.items():
            published_df2.loc[:, '都市計画'] = published_df2.loc[:, '都市計画'].str.replace(key,value)

        published_df2['logy'] = np.log1p(published_df2['y'])
        published_df2 = published_df2.rename(columns={'利用の現況': '用途'})
        
        return published_df2

    def all(self) -> pd.DataFrame:
        self.df = self.rename()
        self.df = self.area()
        self.df = self.create_features()
        self.df = self.create_new_df()

        return self.df