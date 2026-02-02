from flask import Flask, render_template, request, jsonify
import joblib

# 載入模型
model = joblib.load('model.joblib')

# 創建Flask應用程序
app = Flask(__name__)

# 創建路由，用於處理HTTP GET請求
@app.route('/', methods=['GET'])
def home():
    return render_template('q1.html')




# 創建路由，用於處理HTTP POST請求
@app.route('/submit', methods=['POST'])
def submit():
    #name = request.form['name']
    #q1 = int(request.form['q4'])
    q2 = int(request.form['q2'])#第一題
    q3 = int(request.form['q3'])
    q4 = int(request.form['q4'])
    q5 = int(request.form['q5'])
    q6 = int(request.form['q6'])
    q7 = int(request.form['q7'])
    q8 = int(request.form['q8'])
    q9 = int(request.form['q9'])
    q10 = int(request.form['q10'])
    q11= int(request.form['q11'])
    q12 = int(request.form['q12'])
    q13 = int(request.form['q13'])
    q14 = int(request.form['q14'])
    q15 = int(request.form['q15'])
    
    
    
    data1 =[[q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15]]
    input_data=data1
        
    
    #return data1




    # 從HTML表單中獲取數據
    #input_data = request.form['input_data']
    #print(input_data)
    # 使用模型進行預測
    prediction = model.predict(input_data)

    if   prediction.tolist()==['x']:
        #result = {'prediction': "資管"}
        result="資訊管理學系"
        return render_template('result1.html', prediction=result)
    if   prediction.tolist()==['y']:
        result = {'prediction': "財金"}
        return render_template('result2.html', prediction=result)
    if   prediction.tolist()==['z']:
        result = {'prediction': "國企"}
        return render_template('result3.html', prediction=result)
    







    # 將NumPy array轉換為JSON格式的字串
    #result = {'prediction': prediction.tolist()}
    
    # 返回JSON格式的字串
    #return jsonify(result)
    #return  prediction

    # 返回預測結果到HTML頁面
    #return render_template('result.html', prediction=result)



# 啟動Flask應用程序
if __name__ == '__main__':
    app.run(debug=True)
