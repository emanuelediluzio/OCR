import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import os

# --- CONFIGURAZIONE ---
# Se lasci vuoto IMAGE_PATH, lo script cercherà automaticamente
# la prima immagine valida nella cartella corrente.
# Attenzione: su Windows, se scrivi un percorso a mano, usa:
# - una stringa raw: r"C:\percorso\file.webp"
#   oppure
# - le doppie backslash: "C:\\percorso\\file.webp"
IMAGE_PATH = r"C:\Users\emanuele.diluzio\OneDrive - PROMETEIA SPA\Desktop\d.webp"  # es. "documento_prova.jpg" oppure lascia vuoto
MODEL_PATH = "zai-org/GLM-OCR"

# Formati supportati (Pillow deve essere installato con supporto webp)
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]


def main():
    print("--- Avvio GLM-OCR ---")

    # 1. Risoluzione/controllo immagine
    image_path = IMAGE_PATH.strip()

    # Se non è stato impostato un percorso, cerca automaticamente
    # la prima immagine valida nella cartella corrente.
    if not image_path:
        files = os.listdir(".")
        candidates = [
            f for f in files
            if os.path.isfile(f)
            and os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ]
        if not candidates:
            print("ERRORE: Nessuna immagine trovata nella cartella corrente.")
            print(f"Formati supportati: {', '.join(SUPPORTED_EXTENSIONS)}")
            return
        # Prende la prima immagine trovata
        image_path = candidates[0]
        print(f"Nessuna immagine specificata, uso automaticamente: {image_path}")
    else:
        if not os.path.exists(image_path):
            print(f"ERRORE: Non trovo l'immagine al percorso: {image_path}")
            print("Modifica la variabile 'IMAGE_PATH' nello script.")
            return
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            print(f"ERRORE: formato file non supportato: {ext}")
            print(f"Formati supportati: {', '.join(SUPPORTED_EXTENSIONS)}")
            return

    # 2. Rilevamento Dispositivo (Hardware) – usato solo per log,
    # il mapping reale del dispositivo è gestito da transformers (device_map="auto")
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"  # Nvidia GPU
    elif torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon (Mac M1/M2/M3)

    print(f"Dispositivo rilevato: {device.upper()}")

    # 3. Caricamento Processor e Modello
    print("Caricamento del modello in corso... (potrebbe richiedere qualche minuto la prima volta)")

    try:
        processor = AutoProcessor.from_pretrained(MODEL_PATH)

        # Seguiamo l'esempio ufficiale Hugging Face per GLM-OCR
        model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=MODEL_PATH,
            torch_dtype="auto",
            device_map="auto",
        )

        model.eval()
        print("Modello caricato con successo!")

    except Exception as e:
        print(f"\nERRORE CRITICO nel caricamento del modello:\n{e}")
        return

    # 4. Esecuzione OCR
    print(f"\nAnalisi dell'immagine: {image_path}...")

    # Usiamo ancora PIL per assicurarci che il file sia apribile,
    # ma l'immagine viene passata al modello tramite il percorso nel messaggio.
    _ = Image.open(image_path).convert("RGB")

    # Prompt standard/document parsing: "Text Recognition:"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": "Text Recognition:"},
            ],
        }
    ]

    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        # Alcune versioni del processor possono restituire token_type_ids,
        # che GLM-OCR non usa.
        inputs.pop("token_type_ids", None)

        generated_ids = model.generate(**inputs, max_new_tokens=8192)

        # Decodifica solo i token generati, escludendo il prompt
        output_text = processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=False,
        )

        print("\n" + "=" * 40)
        print("RISULTATO OCR:")
        print("=" * 40 + "\n")
        print(output_text)
        print("\n" + "=" * 40)

        # Salvataggio del risultato OCR su file di testo
        base_name, _ = os.path.splitext(os.path.basename(image_path))
        output_path = f"{base_name}_ocr.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_text)

        print(f"\nRisultato OCR salvato in: {output_path}")

    except Exception as e:
        print(f"Errore durante l'inferenza: {e}")


if __name__ == "__main__":
    main()

