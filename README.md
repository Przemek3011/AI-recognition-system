# AI-recognition-system
Model AI który rozpoznaje 10 fraz. Projekt edukacyjny, w którym zwrócimy uwagę na metody niszczenia modeli AI (injections) i przeprowadzenie postmortem analysis. 
# Cel-Projektu
Projekt edukacyjny poświęcony tworzeniu i testowaniu bezpieczeństwa modeli AI. 
# Etapy-Projektu
1. Stworzenie modelu
   - Znalezienie bazy danych z próbkami, do nauki modelu lub stworzenie tych danych za pomocą narzędzi TTS.
   - Wytrenowanie modelu za pomocą biblioteki pytorch lub tensorflow.
   - Analiza skutecznosci modelu.
2. Testowanie modelu
   - Testowanie bezpieczeństwa modelu (artykuły OWASP: https://genai.owasp.org/llm-top-10)
   - Biblioteki i Frameworki do testowania modelu (np. https://github.com/Trusted-AI/adversarial-robustness-toolbox)
3. Analiza postmortem i naprawa zainfekowanego modelu
   - Narzędzia do analizy postmortem (np. https://github.com/evidentlyai/evidently)

