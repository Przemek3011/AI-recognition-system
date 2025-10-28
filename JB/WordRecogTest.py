import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, random_split, Dataset
import os
import random

# ====================================================================
# 0. KONFIGURACJA
# ====================================================================

# 10 słów docelowych (z Google Speech Commands Dataset)
TARGET_CLASSES = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
NUM_CLASSES = len(TARGET_CLASSES)

# Parametry Audio
SAMPLE_RATE = 16000 # Częstotliwość próbkowania
N_MELS = 64         # Liczba pasków Mel w spektrogramie
MAX_FRAMES = 100    # Docelowa długość spektrogramu (100 klatek)

# Parametry Treningu
DATA_PATH = './speech_commands_data'
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {DEVICE}")

# Globalne mapowanie etykiet (do użytku w collate_fn)
label_to_index = {label: i for i, label in enumerate(TARGET_CLASSES)}
index_to_label = {v: k for k, v in label_to_index.items()}


# ====================================================================
# 1. ARCHITEKTURA MODELU (SimpleAudioClassifier)
# ====================================================================

class SimpleAudioClassifier(nn.Module):
    """Prosta sieć CNN do klasyfikacji Mel Spectrogramów."""
    def __init__(self, num_classes):
        super(SimpleAudioClassifier, self).__init__()
        
        # Warstwa Konwolucyjna 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) # Redukcja do 32x50
        
        # Warstwa Konwolucyjna 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) # Redukcja do 16x25

        # Wejście do warstwy liniowej: 64 kanały * 16 mels * 25 ramek = 25600
        self.fc1_input_features = 64 * 16 * 25 
        
        # Warstwy Liniowe
        self.fc1 = nn.Linear(self.fc1_input_features, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Konwolucja -> ReLU -> BatchNorm -> Pooling
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Wypłaszczanie
        x = x.view(x.size(0), -1)
        
        # Liniowe
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ====================================================================
# 2. PRZETWARZANIE AUDIO I DATALOADERY
# ====================================================================

class AudioTransform:
    """Klasa do konwersji waveform na standaryzowany Mel Spectrogram."""
    def __init__(self, sample_rate=SAMPLE_RATE, n_mels=N_MELS, max_frames=MAX_FRAMES):
        self.max_frames = max_frames
        
        self.transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=400,
            hop_length=160
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def __call__(self, waveform):
        # 1. Generowanie spektrogramu
        spectrogram = self.transform(waveform)
        spectrogram = self.amplitude_to_db(spectrogram)
        
        # 2. Standaryzacja długości (przycinanie lub wypełnianie zerami)
        current_frames = spectrogram.shape[2]
        if current_frames > self.max_frames:
            spectrogram = spectrogram[:, :, :self.max_frames]
        elif current_frames < self.max_frames:
            padding = (0, self.max_frames - current_frames)
            spectrogram = torch.nn.functional.pad(spectrogram, padding, "constant", 0)
            
        # Kształt: [1, N_MELS, max_frames]
        return spectrogram.squeeze(0).unsqueeze(0) 


def collate_fn(batch):
    """
    Funkcja używana przez DataLoader do zbierania próbek w batch.
    Przetwarza surowe waveformy na spektrogramy.
    """
    audio_transform = AudioTransform()
    
    spectrograms = []
    labels = []
    
    for waveform, _, label, _, _ in batch:
        # Pamiętaj: SPEECHCOMMANDS daje (waveform, sr, label, ...)
        
        # Wymagany resamplling, jeśli dane zostały pobrane z innym SR (GSC jest 16k, więc domyślnie to się nie uruchomi)
        if waveform.shape[0] != 1 or waveform.shape[1] < 1:
             # Uproszczone obejście dla ewentualnych uszkodzonych lub pustych plików
             continue 

        # Transformacja waveform do spektrogramu
        spec = audio_transform(waveform)
        spectrograms.append(spec)
        
        # Konwersja etykiety tekstowej na indeks liczbowy
        labels.append(label_to_index[label])

    # Łączenie w Tensor
    if not spectrograms:
        # Pusty batch
        return None, None 

    spectrograms = torch.stack(spectrograms)
    labels = torch.tensor(labels)
    
    return spectrograms, labels


def load_data(root_path):
    """Pobiera, filtruje i dzieli Google Speech Commands Dataset."""
    print("Pobieranie/Wczytywanie Google Speech Commands Dataset...")
    # Pobieranie/wczytywanie danych z torchaudio
    full_dataset = torchaudio.datasets.SPEECHCOMMANDS(root=root_path, download=True)
    
    # Filtrowanie tylko do docelowych klas
    filtered_data = [
        item for item in full_dataset 
        if item[2] in TARGET_CLASSES
    ]
    
    # 80/10/10 podział
    train_size = int(0.8 * len(filtered_data))
    val_size = int(0.1 * len(filtered_data))
    test_size = len(filtered_data) - train_size - val_size

    # Używamy torch.Generator dla powtarzalności podziału
    generator = torch.Generator().manual_seed(42)
    train_data, val_data, test_data = random_split(
        filtered_data, [train_size, val_size, test_size], generator=generator
    )
    
    # Tworzenie DataLoaderów
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f"Dane wczytane. Trening: {len(train_data)}, Walidacja: {len(val_data)}, Test: {len(test_data)}")
    return train_loader, val_loader, test_loader


# ====================================================================
# 3. PĘTLA TRENINGOWA I EWALUACJA
# ====================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Przeprowadza jedną epokę treningową."""
    model.train()
    running_loss = 0.0
    
    for inputs, labels in dataloader:
        if inputs is None: continue # Pomijanie uszkodzonych batchów

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate_model(model, dataloader, device):
    """Ocenia model pod kątem dokładności (accuracy)."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            if inputs is None: continue # Pomijanie uszkodzonych batchów
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0.0
    return accuracy

def train_model(model, train_loader, val_loader, criterion, optimizer, device):
    """Główna funkcja treningowa."""
    best_accuracy = 0.0
    
    print("\n--- Rozpoczynam Trening ---")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_accuracy = evaluate_model(model, val_loader, device)

        print(f"Epoka {epoch}/{NUM_EPOCHS}: | Strata Treningowa: {train_loss:.4f} | Dokładność Walidacyjna: {val_accuracy:.2f}%")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_speech_commands_model.pth')
            print("  -> Model zapisany!")
            
    print(f"\n--- Trening Zakończony ---")
    return best_accuracy

# ====================================================================
# 4. FUNKCJA ROZPOZNAWANIA POJEDYNCZEGO SŁOWA
# ====================================================================

def recognize_single_word(filepath, model, device):
    """Rozpoznaje słowo z pojedynczego pliku audio .wav."""
    
    # Wczytanie modelu, jeśli nie jest załadowany
    if not isinstance(model, SimpleAudioClassifier):
         model = SimpleAudioClassifier(NUM_CLASSES).to(device)
         try:
            model.load_state_dict(torch.load('best_speech_commands_model.pth'))
            print("Wczytano zapisane wagi modelu.")
         except FileNotFoundError:
            return "Błąd: Brak zapisanego modelu do wczytania."

    audio_transform = AudioTransform() 
    
    try:
        # Wczytanie pliku
        waveform, sr = torchaudio.load(filepath)
        
        # Resampling, jeśli to konieczne (dane GSC są 16kHz)
        if sr != SAMPLE_RATE:
            resampler = T.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        # Przetworzenie na spektrogram
        spectrogram_tensor = audio_transform(waveform)
    
        # Dodanie wymiaru batcha: [1, 1, N_MELS, max_frames]
        input_data = spectrogram_tensor.unsqueeze(0).to(device) 
        
        # Predykcja
        model.eval()
        with torch.no_grad():
            output = model(input_data)
            
        # Wybór klasy
        _, predicted_index = torch.max(output.data, 1)
        
        recognized_word = index_to_label[predicted_index.item()]
        
        return f"Rozpoznane słowo: **{recognized_word}**"

    except Exception as e:
        return f"Wystąpił błąd podczas rozpoznawania pliku: {e}"


# ====================================================================
# 5. GŁÓWNA BLOKADA URUCHOMIENIOWA
# ====================================================================

if __name__ == '__main__':
    # 1. Wczytanie Danych
    train_loader, val_loader, test_loader = load_data(DATA_PATH)
    
    # 2. Inicjalizacja Modelu i Narzędzi
    model = SimpleAudioClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Trening
    best_val_acc = train_model(model, train_loader, val_loader, criterion, optimizer, DEVICE)
    
    # 4. Ocena Końcowa
    print("-" * 30)
    model.load_state_dict(torch.load('best_speech_commands_model.pth'))
    test_accuracy = evaluate_model(model, test_loader, DEVICE)
    print(f"Finalna dokładność na zbiorze testowym: **{test_accuracy:.2f}%**")
    print("-" * 30)
    
    # 5. Demonstacja Rozpoznawania (Wymaga pliku .wav!)
    # Aby przetestować, musisz podać ścieżkę do JEDNEGO pliku .wav (np. 'data/down/0a9f9f1b_nohash_0.wav')
    # Znajdź plik w folderze 'speech_commands_data' po pierwszym uruchomieniu.
    
    # >>> PRZYKŁAD ROZPOZNANIA <<<
    # Zalecam znalezienie dowolnego pliku z folderu 'speech_commands_data/down' lub '.../up'
    # i zastąpienie poniższej ścieżki
    try:
        random_file = random.choice(test_loader.dataset.indices)
        test_file_path = test_loader.dataset.dataset.dataset[random_file][0]
        print(f"Test rozpoznawania na losowym pliku testowym: {test_file_path}")
        result = recognize_single_word(test_file_path, model, DEVICE)
        print(result)
    except Exception:
        print("Aby przetestować rozpoznawanie, zmień ścieżkę w kodzie na ścieżkę do rzeczywistego pliku .wav ze zbioru danych.")