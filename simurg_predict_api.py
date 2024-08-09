import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModelForSequenceClassification, pipeline
import tensorflow as tf

# GPU configuration
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass 
app = FastAPI()

# Hugging Face model directory (NER için model dizini)
ner_model_name = "korkmazemin1/Named_entity_recognition_turkish_simurg"
# Hugging Face model directory (Duygu analizi için model dizini)
sentiment_model_name = "korkmazemin1/sentiment_analys_turkish_simurg"

# NER modeli ve tokenizer'ı yükle
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)

# Duygu analizi modeli ve tokenizer'ı yükle
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)

# NER ve duygu analizi pipeline'larını oluştur
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")
sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)

# Veri modeli tanımla
class Item(BaseModel):
    text: str = Field(..., example="""Fiber 100mb SuperOnline kullanıcısıyım yaklaşık 2 haftadır @Twitch @Kick_Turkey gibi canlı yayın platformlarında 360p yayın izlerken donmalar yaşıyoruz. Başka hiç bir operatörler bu sorunu yaşamazken ben parasını verip alamadığım hizmeti neden ödeyeyim ? @Turkcell """)

@app.post("/predict/", response_model=dict)
async def predict(item: Item):
    test_sentence = item.text
    
    test_sentence_cap = test_sentence.capitalize()

    # Metindeki organizasyonları tespit et
    ner_results = ner_pipeline(test_sentence_cap)

    # ## tokenlerini birleştirerek orijinal metindeki kelimeyi elde et
    combined_entities = []
    for entity in ner_results:
        if entity['word'].startswith('##') and combined_entities:
            # Orijinal metinden birleşik kelimeyi al
            combined_start = combined_entities[-1]['start']
            combined_end = entity['end']
            full_word = test_sentence[combined_start:combined_end]
            combined_entities[-1]['word'] = full_word
            combined_entities[-1]['end'] = combined_end
        else:
            combined_entities.append(entity)

    # @ sembollerini organizasyon olarak ekle
    tokens = test_sentence.split()
    for idx, token in enumerate(tokens):
        if token.startswith('@'):
            start = test_sentence.find(token)
            end = start + len(token)
            combined_entities.append({
                'entity_group': 'ORG',
                'word': token,
                'start': start,
                'end': end
            })

    # Organizasyonları filtrele
    organizations = [entity for entity in combined_entities if entity['entity_group'] == 'ORG']

    # Organizasyonları başlangıç pozisyonlarına göre sırala
    organizations = sorted(organizations, key=lambda x: x['start'])

    # Organizasyonlar etrafındaki duygu durumunu analiz et
    context_window_size = 10  # Organizasyonlar etrafındaki kelime sayısı

    entity_list = []
    results = []

    i = 0
    while i < len(organizations):
        org = organizations[i]
        start_idx = org['start']
        end_idx = org['end']

        # Yakındaki diğer organizasyonları hesaba kat
        while i + 1 < len(organizations) and (organizations[i + 1]['start'] - end_idx <= 1):
            end_idx = organizations[i + 1]['end']
            i += 1

        # Context penceresini tanımla (Sadece sağa bakacak şekilde ayarlıyoruz)
        context_start = end_idx
        context_end = min(len(test_sentence), end_idx + context_window_size)
        context = test_sentence[context_start:context_end]
        
        # Duygu analizi yap
        sentiment = sentiment_pipeline(context)
        
        # Sonuçları topla
        entity_list.append(org['word'])
        sentiment_label = sentiment[0]['label']
        # Duygu durum etiketlerini metne dönüştür
        if sentiment_label == 'Negative':
            sentiment_text = "olumsuz"
        elif sentiment_label == 'Neutral':
            sentiment_text = "nötr"
        elif sentiment_label == 'Positive':
            sentiment_text = "olumlu"
        else:
            sentiment_text = "nötr"  # Beklenmeyen etiketler için varsayılan durum
            
        results.append({
            "entity": org['word'],
            "sentiment": sentiment_text
        })
        
        i += 1

    # Belirtilen formatta çıktı oluştur
    output = {
        "entity_list": entity_list,
        "results": results
    }

    return output

@app.get("/", response_class=HTMLResponse)
async def get():
    html_content = """
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FastAPI Arayüzü</title>
    </head>
    <body>
        <h1>Metin Analizi</h1>
        <form id="analysisForm">
            <label for="text">Metin:</label><br>
            <textarea id="text" name="text" rows="4" cols="50"></textarea><br><br>
            <input type="submit" value="Analiz Et">
        </form>
        <h2>Sonuçlar:</h2>
        <pre id="results"></pre>

        <script>
            document.getElementById('analysisForm').addEventListener('submit', async function(event) {
                event.preventDefault();
                const text = document.getElementById('text').value;
                const response = await fetch('/predict/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ text })
                });
                const result = await response.json();
                document.getElementById('results').textContent = JSON.stringify(result, null, 2);
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7444)
    #http://127.0.0.1:8000/
