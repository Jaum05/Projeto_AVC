python -m http.server 5500 --bind 127.0.0.1 (RODAR O SERVIDOR, NÃO ACESSAR PELO TERMINAL ACESSE PELO HTTP: ABAIXO)

python -m uvicorn api_avc:app --reload --app-dir .\src  (RODAR A API)

http://localhost:5500/Front/index.html (QUANDO A API ESTIVER RODANDO E O SERVIDOR TAMBÉM, ACESSE ESTE LINK OU PELO GO LIVER SERVER NO INDEX.HTML :))

SE NÃO ME ENGANO TEM QUE INSTALAR MAIS ESSAS BIBLIOTECAS

pip install imblearn xgboost joblib (deve ter mais) 
