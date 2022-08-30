import joblib
import pandas as pd
import numpy as np
import re
# 사전에 학습된 모델을 이용한 예측 기능 제공
def predict( src ):
    '''
        예측 담당 함수(머신러닝/딥러닝 기술으로 규칙을 완성)
        src    : 입력, 예측 모델이 예측을 수행할수 있는 형태로 제공, DataFrame, ( 개수, 65536 )
        return : 출력, 언어를 예측하여 국가코드를 리턴, ndarray, ( 문자열정답,  )
    '''
    # 1. 모델 로드 ( 엔트리포인트기준에서 상대경로를 설정 or 절대경로계산하여설정 )
    model   = joblib.load('ml.pkl')
    # 2. 예측
    # *.npy 파일 로드해서 이데이터를 입력으로 하여 예측 수행    
    y_pred  = model.predict( src )
    # 3. 응답구성및 리턴
    return y_pred

# 데이터 전처리 기능 제공
def preprocessing( oriText=None ):
    if not oriText:
        return pd.DataFrame( np.load( './model/test_features.npy') )
    else:
        '''
            # 한국어와 일본어등 고유한 문자를 쓰는 국가들은 굳이 학습하지 않아도 규칙성으로 해결가능
            # 전체 문자대비 영어(한국어, 일본어)의 비중이 80%이상이면 알파벳으로 인지 설정
            # 입력 텍스트를 3개의 정규식으로 각각 처리후 각문자별 위치의 카운트를 계산해서, 원문자개수대비
            # 80%이상이면 해당 문자로 분류하겠다(프로그래밍 규칙)
            # 이중 알파벳으로 판독된것은 모델로 예측 수행 => en, fr, tl, id 분류
            # 한국어, 일본어는 그대로 결론 도달함
        '''
        # 1. 텍스트 소문자 처리
        oriText = oriText.lower().strip()   
        # 2. 원본 문자열의 총 문자수 기록
        oriTextSize = len(oriText)
        # 3. 정규식 3개 준비
        alpha  = re.compile('[^a-zA-Z]*')
        hangul = re.compile('[^가-힣ㅏ-ㅣㄱ-ㅎ]*')
        japan  = re.compile('[^ぁ-ゔァ-ヴー々〆〤]*')
        # 4. 원문을 정규식 처리해서 카운트를 담는다
        a_txt  = alpha.sub('', oriText) 
        h_txt  = hangul.sub('', oriText) 
        j_txt  = japan.sub('', oriText) 
        # 5. 개별문자수/전체원본문자수 비중에 80%이상일경우 해당문자로 분류
        if len(a_txt) / oriTextSize >= 0.8:
            STD_LEN = 2**16
            STD_LEN_LAST_INDEX = STD_LEN-1
            counts      = np.zeros( STD_LEN )
            # 6. 빈도 계산
            for ch in a_txt:
                no            = ord(ch)
                if no > STD_LEN_LAST_INDEX: # 특정 문자의 방번호가 65535번을 넘어섰다
                    continue
                counts[ no ] += 1
            # 7. 정규화
            count_norms = counts / len(a_txt)
            # 8. shape 변경 1D (65536,) => 2D (1, 65536)
            return count_norms.reshape( ( 1, STD_LEN)  )
            
        elif len(h_txt) / oriTextSize >= 0.8:
            print('한국어', len(h_txt) / oriTextSize)
        else:
            print('일본어', len(j_txt) / oriTextSize)
        

# 전체 구동 메인
def predictMainStream( oriText=None ):
    if not oriText:
        text = preprocessing()        
    else:
        # 실제 예측이 처리
        text = preprocessing( oriText )
    return predict( text )

# 단위 테스트용
if __name__ == '__main__':
    #print(  predictMainStream() )
    print(  predictMainStream('あの日の悲しみといったら、忘れられません。') )
    pass