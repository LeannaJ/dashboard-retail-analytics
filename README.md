# 🛒 Dunnhumby Retail Analytics Dashboard

**Data Analyst Portfolio Project** - Consumer Goods Retail Analytics

## 📋 프로젝트 개요

이 프로젝트는 Dunnhumby의 소비재 회사 synthetic data를 활용하여 종합적인 리테일 분석 대시보드를 구축한 Data Analyst 포트폴리오 프로젝트입니다.

## 🎯 주요 기능

### 📈 비즈니스 개요
- 실시간 매출 및 거래 지표 모니터링
- 시계열 매출 분석 (일별/주별/월별)
- 요일별 및 시간대별 매출 패턴 분석

### 👥 고객 분석
- **RFM 분석**: Recency, Frequency, Monetary 점수 기반 고객 세분화
- 고객 인구통계학적 분석 (연령대, 거주 유형별)
- 고객 생애 가치(CLV) 계산 및 분석
- 코호트 분석을 통한 고객 유지율 분석

### 🛍️ 상품 분석
- 상품 카테고리별 매출 분석
- 브랜드 성과 분석
- 상품 성과 매트릭스 (빈도 vs 평균 매출)
- 부서별 매출 분포

### 🎯 캠페인 분석
- 캠페인 타입별 효과성 분석
- 캠페인 참여 vs 비참여 고객 비교
- 캠페인 ROI 분석

### 📊 고급 분석
- 코호트 분석
- 상관관계 분석
- 간단한 예측 모델링 (매출 트렌드 예측)

## 🛠️ 기술 스택

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Language**: Python 3.8+

## 📊 데이터셋 구조

### 주요 테이블
1. **transaction_data.csv** - 고객 구매 거래 데이터
2. **hh_demographic.csv** - 고객 인구통계학적 정보
3. **product.csv** - 상품 정보 (브랜드, 카테고리, 부서 등)
4. **campaign_table.csv** - 캠페인 참여 고객 정보
5. **campaign_desc.csv** - 캠페인 설명 및 기간
6. **coupon_redempt.csv** - 쿠폰 사용 내역
7. **coupon.csv** - 쿠폰 정보

### 주요 컬럼
- **거래 데이터**: household_key, BASKET_ID, PRODUCT_ID, SALES_VALUE, QUANTITY
- **고객 데이터**: 연령대, 거주 유형, 자녀 유무 등
- **상품 데이터**: 브랜드, 카테고리, 부서, 상품 설명
- **캠페인 데이터**: 캠페인 타입, 참여 고객, 기간

## 🚀 실행 방법

### 1. 환경 설정
```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2. 애플리케이션 실행
```bash
streamlit run Streamlit_Retail.py
```

### 3. 브라우저에서 접속
- 기본 URL: http://localhost:8501

## 📈 주요 분석 인사이트

### 1. 매출 트렌드
- 시계열 분석을 통한 매출 패턴 파악
- 계절성 및 트렌드 분석

### 2. 고객 세분화
- RFM 분석을 통한 고객 등급별 특성 파악
- 고가치 고객 식별 및 타겟팅 전략 수립

### 3. 상품 성과
- 카테고리별 매출 기여도 분석
- 브랜드별 성과 비교

### 4. 캠페인 효과
- 마케팅 캠페인의 실제 ROI 측정
- 효과적인 캠페인 타입 식별

## 🎨 대시보드 특징

### 인터랙티브 필터링
- 날짜 범위 선택
- 고객 세그먼트 필터링
- 상품 카테고리 필터링

### 다양한 시각화
- 동적 차트 (Plotly)
- 히트맵 및 상관관계 분석
- 코호트 분석 히트맵

### 반응형 디자인
- 모바일 친화적 레이아웃
- 직관적인 네비게이션

## 📊 비즈니스 가치

### 1. 데이터 기반 의사결정
- 실시간 비즈니스 지표 모니터링
- 정량적 분석을 통한 전략 수립

### 2. 고객 이해도 향상
- 고객 행동 패턴 분석
- 개인화된 마케팅 전략 수립

### 3. 운영 효율성
- 상품 성과 최적화
- 재고 관리 개선

### 4. 마케팅 ROI 향상
- 캠페인 효과 측정 및 최적화
- 타겟 마케팅 전략 수립

## 🔧 커스터마이징

### 새로운 분석 추가
1. `main()` 함수 내 새로운 탭 추가
2. 해당 분석 함수 구현
3. 시각화 및 인사이트 추가

### 데이터 소스 변경
1. `load_data()` 함수 수정
2. 데이터 전처리 로직 조정
3. 컬럼 매핑 업데이트

## 📝 라이선스

이 프로젝트는 포트폴리오 목적으로 제작되었습니다.

## 🤝 기여

이 프로젝트에 대한 피드백이나 개선 제안은 언제든 환영합니다!

---

**Built with ❤️ for Data Analytics Portfolio**
