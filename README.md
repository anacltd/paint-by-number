# paint-by-number
🖌️ Streamlit app to generate a paint-by-number version of a picture.

<img width="1479" height="756" alt="Screenshot 2025-08-25 at 09 38 04" src="https://github.com/user-attachments/assets/e4aecb15-5a11-4012-8e8d-047a0d2a003b" />

## How to use
Per usual, clone the repo, create a virtual environment and install the project:
```bash
git clone https://github.com/anacltd/paint-by-number.git
cd paint-by-number
python3.10 -m venv venv
source venv/bin/activate
pip install -e .
```

To launch the Streamlit app:
```bash
streamlit run cd src/paint_by_number/app.py
```

Clicking `Download results` will download a ZIP file containing the colour-blocked image, the colour palette and the paint-by-number version.

That's it!

## Nota bene
The paint-by-number version of the picture is not 100% accurate, especcially if the picture has many tiny details.
