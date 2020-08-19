import pandas as pd

res1 = pd.read_csv("../output/result_realestate_202008010_01.csv")
#res2 = pd.read_csv("../output/result_realestate_202008010_03.csv")
res2 = pd.read_csv("../output/result_realestate_20200809_03.csv")
res3 = pd.read_csv("../output/result_realestate_20200809_04.csv")
score1 = 0.26270
score2 = 0.26675
score3 = 0.26707

res_omikuji = pd.DataFrame()
res_omikuji["id"] = res1["id"]
res_omikuji["y"] = res1["y"]*(score1/(score1+score2+score3)) + res2["y"]*(score2/(score1+score2+score3)) #+ res3["y"]*(score3/(score1+score2+score3))
#display(res1.head())
res_omikuji.head()
res_omikuji.to_csv("../output/res_omikuji2.csv", index=False)