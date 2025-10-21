# AI-recognition-system
Model AI który rozpoznaje 10 fraz. Projekt edukacyjny, w którym zwrócimy uwagę na metody testowania bezpieczeństwa modeli AI (injections) i przeprowadzenie analizy postmortem. 
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
   - Próba ataku(injekcji) na model
3. Analiza postmortem i naprawa zainfekowanego modelu
   - Narzędzia do analizy postmortem (np. https://github.com/evidentlyai/evidently)

# Technologie
- **PyTorch** *Biblioteka do trenowania modelu ze wsparciem audio*
- **Scikit-learn** *Analiza precyzji modelu i wizualizacja*
- **gTTS** *Generacja próbek głosowych*
- **Evidently** *Analiza postmortem modelu*
- **Google Speech Commands DataSet** *Dane do uczenia modelu*

# Podział prac
1. Model & Data
   - Architektura modelu
   - Wyszukiwanie/generacja zbioru danych
   - Preprocessing audio
2. Security
   - Testy bezpieczeństwa
   - Przeprowadzenie ataków/injekcji
   - Analiza możliwości ulepszenia modelu
3. Analysis
   - Analiza precyzji modelu
   - Wizualizacja wyników
   - Analiza postmortem

# Harmonogram
#### Iteracja 1(pierwsze dwa tygodnie)
Zaprojektowanie modelu, zbiór danych, przygotowywanie do testów i ataków

- Wspólne opracowanie architektury modelu
- Wyszukanie i przygotowanie zbioru danych (Google Speech Commands / TTS)
- Ustalenie fraz do rozpoznawania
- Przygotowanie środowiska do analizy (Evidently)
- Projekt szablonu testów i wizualizacji wyników

#### Iteracja 2 (Tyg. 3-5)
Implementacja pierwszej wercji projektu, przeprowadzenie ataków
1.
- Implementacja i trenowanie modelu 
- Ekstrakcja cech audio
- Ewaluacja skuteczności modelu (accuracy, loss) 
2.
- Przygotowanie środowiska testowego
- Implementacja pierwszych ataków
- Testy wpływu ataków na wczesną wersję modelu 
3.
- Analiza wyników modelu i wizualizacja metryk 
- Porównanie skuteczności między wersjami modelu  
- Przygotowanie wstępnych raportów analitycznych

#### Iteracja 3 (Tyg. 6-7)
Próba udoskonalenia modelu i zabezpieczenia przed injekcjami
1.
- Retraining modelu po atakach 
- Udoskonalenie architektury i augmentacja danych 
- Analiza stabilności modelu 
2.
- Rozszerzenie testów bezpieczeństwa (PGD, poisoning)
- Implementacja metod obron (adversarial training) 
- Ocena skuteczności mechanizmów ochrony 
3.
- Analiza postmortem po atakach 
- Wykrywanie driftu modelu i zmian w danych  
- Wizualizacja porównawcza „przed” i „po” ataku

#### Iteracja 4 (Tyg. 8-9)
Analiza wyników, doprecyzowanie i optymalizacja modelu
1.
- Optymalizacja finalnego modelu i zapis wersji produkcyjnej  
- Wsparcie przy integracji wyników analitycznych 
2.
- Opracowanie końcowego raportu bezpieczeństwa 
- Podsumowanie wszystkich testów i podatności 
3.
- Opracowanie końcowej analizy i wniosków  
- Stworzenie prezentacji projektu i raportu końcowego
