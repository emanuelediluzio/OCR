# OCR – GLM-OCR Wrapper

Wrapper semplice in Python per usare il modello **[`zai-org/GLM-OCR`](https://huggingface.co/zai-org/GLM-OCR)** in locale per fare OCR avanzato (anche su layout complessi) da riga di comando.

Supporta:
- **immagini singole** (percorso assoluto o file nella cartella del progetto)
- vari formati: `jpg`, `jpeg`, `png`, `webp`, `bmp`, `tif`, `tiff`
- **salvataggio automatico** dell’output in un file `*_ocr.txt`

---

## Requisiti

- Python 3.10+ consigliato
- GPU NVIDIA opzionale (funziona anche solo CPU, ma più lento)

### Dipendenze Python

Da terminale, nella cartella del progetto:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# GLM-OCR richiede una versione recente di transformers:
python -m pip uninstall -y transformers
python -m pip install "git+https://github.com/huggingface/transformers.git"
```

> Nota: il warning sui **symlink** di Hugging Face su Windows può essere ignorato,
> riguarda solo l’ottimizzazione della cache.

---

## Uso

### 1. Configurare l’immagine

Nel file `run_ocr.py` c’è la variabile:

```python
IMAGE_PATH = r"C:\percorso\alla\mia\immagine.webp"
```

Su Windows:
- usa una **stringa raw**: `r"C:\cartella\file.webp"`  
  oppure
- raddoppia le backslash: `"C:\\cartella\\file.webp"`.

In alternativa puoi lasciare `IMAGE_PATH` **vuoto**:

```python
IMAGE_PATH = ""
```

In questo caso lo script cercherà automaticamente la **prima immagine valida** nella cartella corrente con estensione tra:

```text
.jpg, .jpeg, .png, .webp, .bmp, .tif, .tiff
```

### 2. Eseguire l’OCR

Da terminale, nella cartella del progetto:

```bash
python run_ocr.py
```

Lo script:
- rileva il dispositivo (`CPU` / `CUDA` / `MPS`, solo log informativo)
- scarica al primo run il modello `zai-org/GLM-OCR` da Hugging Face
- esegue l’OCR usando il prompt `"Text Recognition:"`
- stampa il testo riconosciuto a schermo
- salva il risultato in un file di testo affiancato all’immagine:

```text
<nome_immagine>_ocr.txt
```

Esempio:
- immagine: `fattura.webp`
- output: `fattura_ocr.txt`

---

## Struttura del progetto

- `run_ocr.py` – script principale che:
  - seleziona l’immagine (`IMAGE_PATH` o auto-discovery)
  - carica `AutoProcessor` e `AutoModelForImageTextToText` per `zai-org/GLM-OCR`
  - esegue `generate` e decodifica l’output
  - salva il risultato in `<nome_immagine>_ocr.txt`
- `requirements.txt` – dipendenze di base (transformers viene poi aggiornato da GitHub come indicato sopra)

---

## Note e troubleshooting

- Se vedi errori tipo `Unrecognized configuration class ... GlmOcrConfig`:
  - assicurati di aver installato **transformers da GitHub** come indicato sopra.
- Se l’esecuzione è molto lenta:
  - stai probabilmente usando la **CPU**; con una GPU NVIDIA (`CUDA`) sarà molto più veloce.
- Se compaiono warning su symlink su Windows:
  - sono solo avvisi relativi alla cache, il modello funziona comunque.
