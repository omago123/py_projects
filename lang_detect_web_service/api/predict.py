from flask import jsonify,request



def predict():
    # json으로 응답
    # 1. 클라이언트가 보낸 데이터 획득(post 방식)
    law_text = request.form.get('key')
    print(law_text)
    # 2. 전처리(훈련중일때는 정규식 사용이 이미 답을 알고 있어서 혼선이 없었음, 예측시에는 문제됨)
    #    입력 데이터의 최종 형태
    # 3. 모델에 입력 후 예측 수행
    # 4. 예측 결과를 받아서 응답처리
    return jsonify( {'code':1, 'value':'한국어'})