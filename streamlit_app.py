#ライブラリ読み込み
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

#対数収益率の正規分布を仮定した長期投資シミュレーションモデル
class asset_model:
    #初期化時の処理
    def __init__(self, mu_y, s_y):
        #パーセント表記から少数へ変換
        mu_y = mu_y / 100
        s_y = s_y / 100
        #入力された対数収益率の年次リターン・リスク（標準偏差）を月次へ変換
        self.mu_m = mu_y /12
        self.s_m = s_y / np.sqrt(12)
    
    #対数収益率の長期モンテカルロシミュレーションを実施
    def run_mc(self, dur_y, n):
        #プログレスバー出力
        progress_text = "Monte Carlo Calculation of "+ str(dur_y) + " years in progress."
        my_bar = st.progress(0, text=progress_text)
        #入力された投資期間を年→月へ変換
        dur_m = dur_y * 12
        my_bar.progress(10, text=progress_text) #プログレスバー出力
        # [(試行回数), (投資期間_月)]の正規分布N(μ,σ)に従う乱数行列として毎月の対数収益率を生成
        self.A = np.random.normal(self.mu_m, self.s_m,size=[n,dur_m])        
        
        self.dur_m = dur_m
        my_bar.progress(100, text=progress_text) #プログレスバー出力
        
    #初期投資額x_init, 毎月積立額delta_mを投資した場合の資産推移を計算
    def exercise(self, x_init, delta_m):
        #プログレスバー出力
        progress_text = "Investment simulation in progress."
        my_bar = st.progress(0, text=progress_text)
        #月別の投資額のリスト
        x = delta_m * np.ones(self.dur_m)
        #初期投資額を追加
        x[0] = x[0] + x_init
        
        #総資産額のリスト
        ta_list = []
        #累計投資額のリスト
        capital = [] 
        
        #総資産額の推移を1年毎(12ヶ月毎)に集計する
        for i in range(11, self.dur_m,12):
            #集計時点までの対数収益率行列を切り出し
            A_tmp = self.A[:,:i+1] 
            #集計時点までの投資額のリストを切り出し
            x_tmp = x[:i+1]
            #対数収益率行列の後ろ向き累積和（Suffix sum) 、連続複利ベースの期間終了時点の収益率を計算
            rev_cum = np.cumsum(A_tmp[:, ::-1], axis=1)
            lr_tmp = rev_cum[:, ::-1]
            #対数収益率から、価格変動比へ変換
            r_tmp = np.exp(lr_tmp)
            #各試行の、投資期間経過後の資産額を計算
            ta = np.dot(r_tmp,x_tmp)
            #計算結果を格納
            ta_list.append(ta)
            
            #累計投資額を格納
            capital.append(x_tmp.sum())
            #プログレスバー出力
            my_bar.progress(i/(self.dur_m-1), text=progress_text)
            
        #1年毎の各種統計値を計算
        #平均
        mean = np.mean(ta_list, axis=1)
        #パーセンタイル値（np.percetileは下位○%を入力するが、出力としては上位○%の表記とする）
        p10 = np.percentile(ta_list,90,axis=1)
        p30 = np.percentile(ta_list,70,axis=1)
        p50 = np.percentile(ta_list,50,axis=1)
        p70 = np.percentile(ta_list,30,axis=1)
        p90 = np.percentile(ta_list,10,axis=1)
    
        year = np.arange(self.dur_m/12)+1
        
        #計算結果をDataFrameに格納
        df_result = pd.DataFrame(np.transpose([year,mean,p10,p30,p50,p70,p90,capital]),columns=["Year","Mean","P10","P30","P50","P70","P90","Capital"])

        #グラフ出力
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_result["Year"],y=df_result["P90"],name="P90"))
        fig.add_trace(go.Scatter(x=df_result["Year"],y=df_result["P10"],name="P10", fill="tonexty"))
        fig.add_trace(go.Scatter(x=df_result["Year"],y=df_result["P70"],name="P70"))
        fig.add_trace(go.Scatter(x=df_result["Year"],y=df_result["P30"],name="P30", fill="tonexty"))
        fig.add_trace(go.Scatter(x=df_result["Year"],y=df_result["P50"],name="P50"))
        fig.add_trace(go.Scatter(x=df_result["Year"],y=df_result["Capital"],name="Capital"))
        fig.update_layout(hovermode="x")
        fig.update_layout(title="Result of Simulation", xaxis_title="Year")
        st.plotly_chart(fig, use_container_width=True)
        
        #DataFrame出力
        df_result = df_result.round().astype(int)
        st.dataframe(df_result)
      

st.title("長期積立投資シミュレーション")
st.markdown("""
## 概要
特定のリターン・リスクを持つ資産に、毎月積立投資を行った場合の長期資産推移をモンテカルロ法によりシミュレーションします。

## 使い方
対象資産の年換算リターン・リスク、投資期間・初期投資額・毎月投資額を入力し「Run」ボタンをクリック
## 実行例
 """)
# 期待対数収益率、標準偏差（%,年換算）
mu = st.number_input("Annualized Return(%)", min_value=0.0, max_value=None, value=8.0, step=0.1, help="対象資産の対数収益率（年換算）。対数収益率が不明の場合、収益率を近似値として使用しても可")
s = st.number_input("Annualized Risk(%)", min_value = 0.0, max_value=None, value=20.0, step=0.1, help="対数収益率（年換算）の標準偏差")


# 投資期間（年）
dur_y = st.number_input("Investment Duration（year）", min_value=1, max_value=30, value=10, step=1, help="積立投資を行う期間。Streamlit Cloudのリソース制約上、上限は30年")

# 初期投資額
x_init = st.number_input("Initial investment amount", min_value=0, max_value=None, value=0, step=1, help="初期投資額")
# 毎月積立額
delta_m = st.number_input("Monthly investment amount", min_value=0, max_value=None, value=10, step=1, help="毎月の積立額")

# 試行回数
n = st.number_input("Number of trials for Monte Carlo Calculation", min_value=100, max_value=20000, value=10000, step=1, help="モンテカルロ法による試行回数。Streamlit Cloudのリソース制約上、上限は20000")

if st.button("Run"):
	#リターンmu、リスクsのモデルを定義
	model = asset_model(mu,s)
	#投資期間dur_y、試行回数nのモンテカルロシミュレーションを実行
	model.run_mc(dur_y,n)
	#初期投資x_init、毎月delta_mの積立投資を行った場合の資産推移を計算
	model.exercise(x_init,delta_m)


st.markdown("""
## 解説

Qiita  
https://qiita.com/yasuyuki-s/items/4d22c9ab1f5005b64ae5
 """)