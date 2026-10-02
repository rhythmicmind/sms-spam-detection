"""Fetch UCI SMS Spam Collection; preserves raw messages locally."""
import io, zipfile
from pathlib import Path
from urllib.request import urlopen

if __name__ == '__main__':
    url = 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'
    with urlopen(url, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    target = Path('data/SMSSpamCollection')
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(archive.read('SMSSpamCollection'))
    print(f'Saved {target}; see DATA_CARD.md for attribution and limitations.')
