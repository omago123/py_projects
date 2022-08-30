from flask import Flask, request, render_template, jsonify
# 남이 만들 모듈을 내가 가져다가 사용
from model import predictMainStream

app = Flask(__name__)

@app.route('/')
def home(): 
    # 텍스트를 입력받는 화면 준비해서 클라이언트한테 랜더링하여 제공(서버 사이드 렌더링:SSR)
    return render_template('index.html')


# POST 방식으로 url이 반응하게 허용해야함. 기본은 GET방식임
@app.route('/predict' , methods=['POST'])
def predict():  
    # json으로 응답, 예측행위는 생략(임시) 
    # 1. 클라이언트가 보낸 데이터 획득(post 방식)
    law_text = request.form.get('key')
    print( law_text )
    # 2. 전처리(훈련중일때는 정규식 사용이 이미 답을 알고 있어서 혼선이 없었음, 예측시에는 문제됨)
    #    입력 데이터의 최종 형태 => (1, 65536), DataFrame 형태로 입력
    # 3. 모델에 입력 후 예측 수행
    y_pred = predictMainStream( law_text )
    # 답안지, 결과를 원하는 형태로 변형
    key = y_pred[0]    
    dic = {
        'en':'영어',
        'fr':'프랑스어',
        'tl':'타갈리아어',
        'id':'인도네시아어',
        'ko':'한국어',
        'jp':'일본어',
    }
    # 4. 예측 결과를 받아서 응답처리
    return jsonify( { 'code':1, 'value':dic[ key ] } )

if __name__ == '__main__':
    # 코드를 수정하면 실시간 리로드되어 반영됨
    app.run(debug=True)