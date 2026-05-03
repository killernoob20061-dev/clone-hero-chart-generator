"""
Clone Hero chart scraper — downloads chart+audio pairs from Enchor (Chorus Encore).
Usage: python scraper.py --out ./dataset --limit 2000

API:  POST https://api.enchor.us/search
Download: https://files.enchor.us/{md5}.sng  (self-contained SNG archive)

SNG files contain: notes.chart, song.opus/ogg/mp3, song.ini, album art etc.
Saves each song as: dataset/<md5>/notes.chart + song.opus + meta.json
"""

import os, sys, json, time, argparse, zipfile, struct, re
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ─────────────────────────────────────────────────────────────────────
API_URL     = 'https://api.enchor.us/search'
DL_URL      = 'https://files.enchor.us/{md5}.sng'
RATE_LIMIT  = 0.4
TIMEOUT     = 40
MAX_WORKERS = 4
AUDIO_EXTS  = {'.opus', '.ogg', '.mp3', '.wav'}

# ── Helpers ────────────────────────────────────────────────────────────────────

def safe_filename(s):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', str(s))[:80]

def search_page(query='', page=1):
    """POST one page to the Enchor search API."""
    payload = {
        'search': query, 'page': page,
        'instrument': None, 'difficulty': None,
        'drumType': None, 'drumsReviewed': False,
        'source': 'website'
    }
    try:
        r = requests.post(API_URL, json=payload, timeout=TIMEOUT,
                          headers={'User-Agent': 'CloneHeroTrainer/1.0',
                                   'Content-Type': 'application/json'})
        r.raise_for_status()
        data = r.json()
        return data.get('data', [])
    except Exception as e:
        print(f'  [warn] API page {page}: {e}')
        return []

def iter_all_songs(limit=2000):
    """Yield song dicts until limit reached, paging through all results."""
    seen, page = set(), 1
    while len(seen) < limit:
        songs = search_page(page=page)
        if not songs:
            break
        for s in songs:
            md5 = s.get('md5', '')
            if md5 and md5 not in seen:
                seen.add(md5)
                yield s
                if len(seen) >= limit:
                    return
        print(f'  Page {page}: {len(seen)} songs so far')
        page += 1
        time.sleep(RATE_LIMIT)

# ── SNG parsing ────────────────────────────────────────────────────────────────
# SNG is a simple container: 4-byte magic + file entries
# Format per entry: [uint32 name_len][name][uint64 file_len][data]
SNG_MAGIC = b'SNGPKG\0\0'

def parse_sng(sng_bytes):
    """
    Parse a .sng file and return dict {filename: bytes}.
    SNG format: magic(8) + version(4) + xor_mask(8) + file_count(8) + entries...
    Each entry: name_len(4LE) + name(utf8) + file_len(8LE) + data
    """
    files = {}
    if not sng_bytes.startswith(b'SNGPKG'):
        # Try as zip fallback
        try:
            import io
            with zipfile.ZipFile(io.BytesIO(sng_bytes)) as zf:
                for name in zf.namelist():
                    files[name.lower()] = zf.read(name)
            return files
        except Exception:
            return files

    offset = 6    # skip magic
    # version (uint16)
    offset += 2
    # xor mask (8 bytes) — used to decode file data
    xor_mask = sng_bytes[offset:offset+8]
    offset += 8
    # file count (uint64)
    file_count = struct.unpack_from('<Q', sng_bytes, offset)[0]
    offset += 8

    for _ in range(file_count):
        if offset + 4 > len(sng_bytes):
            break
        name_len = struct.unpack_from('<I', sng_bytes, offset)[0]
        offset += 4
        name = sng_bytes[offset:offset+name_len].decode('utf-8', errors='replace')
        offset += name_len
        file_len = struct.unpack_from('<Q', sng_bytes, offset)[0]
        offset += 8
        data = bytearray(sng_bytes[offset:offset+file_len])
        # XOR decode
        for i in range(len(data)):
            data[i] ^= xor_mask[i % 8]
        files[name.lower()] = bytes(data)
        offset += file_len

    return files

# ── Download & extract ─────────────────────────────────────────────────────────

def download_bytes(url, session):
    try:
        with session.get(url, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            return b''.join(r.iter_content(65536))
    except Exception as e:
        print(f'  [warn] download {url}: {e}')
        return None

def validate_chart(chart_bytes):
    """Chart must have ExpertSingle with at least 50 notes."""
    try:
        text = chart_bytes.decode('utf-8', errors='replace')
        if '[ExpertSingle]' not in text:
            return False
        expert = text.split('[ExpertSingle]')[1].split('}')[0]
        return expert.count(' = N ') >= 50
    except Exception:
        return False

def process_song(song, out_root, session):
    """Download + parse SNG + validate + save one song. Returns folder or None."""
    md5  = song.get('md5', '')
    if not md5:
        return None

    folder = out_root / md5
    if folder.exists() and (folder / 'notes.chart').exists():
        return str(folder)   # already done

    folder.mkdir(parents=True, exist_ok=True)

    # Download SNG
    url  = DL_URL.format(md5=md5)
    data = download_bytes(url, session)
    if not data:
        folder.rmdir()
        return None

    # Parse SNG container
    files = parse_sng(data)
    if not files:
        return None

    # Find chart file
    chart_bytes = None
    for name in ('notes.chart', 'notes.mid'):
        if name in files:
            chart_bytes = files[name]; break
    if chart_bytes is None:
        return None
    if not validate_chart(chart_bytes):
        return None

    # Find audio file
    audio_bytes = audio_ext = None
    for ext in ('.opus', '.ogg', '.mp3', '.wav'):
        for name, content in files.items():
            if name.endswith(ext):
                audio_bytes = content; audio_ext = ext; break
        if audio_bytes:
            break
    if audio_bytes is None:
        return None

    # Save chart
    (folder / 'notes.chart').write_bytes(chart_bytes)

    # Save + convert audio to opus
    raw_audio = folder / f'audio_raw{audio_ext}'
    raw_audio.write_bytes(audio_bytes)
    target = folder / 'song.opus'
    if audio_ext == '.opus':
        raw_audio.rename(target)
    else:
        import subprocess
        subprocess.run(
            ['ffmpeg', '-y', '-i', str(raw_audio), '-c:a', 'libopus',
             '-b:a', '128k', str(target)],
            capture_output=True)
        try: raw_audio.unlink()
        except Exception: pass

    if not target.exists():
        return None

    # Save album art if present
    for name, content in files.items():
        if name.endswith('.jpg') or name.endswith('.png'):
            (folder / 'album.jpg').write_bytes(content)
            break

    # Save metadata
    meta = {
        'md5':     md5,
        'title':   song.get('name', ''),
        'artist':  song.get('artist', ''),
        'album':   song.get('album', ''),
        'genre':   song.get('genre', ''),
        'year':    song.get('year', ''),
        'charter': song.get('charter', ''),
        'length_ms': song.get('song_length', 0),
    }
    (folder / 'meta.json').write_text(json.dumps(meta, indent=2))
    return str(folder)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',     default='./dataset', help='Output directory')
    ap.add_argument('--limit',   type=int, default=2000)
    ap.add_argument('--workers', type=int, default=MAX_WORKERS)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f'Scraping up to {args.limit} songs → {out}')

    songs = list(iter_all_songs(args.limit))
    print(f'Found {len(songs)} songs from Enchor API')
    if not songs:
        print('No songs returned — check your internet connection')
        return

    ok = fail = 0
    session = requests.Session()
    session.headers['User-Agent'] = 'CloneHeroTrainer/1.0'

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_song, s, out, session): s for s in songs}
        for future in as_completed(futures):
            result = future.result()
            if result:
                ok += 1
                if ok % 25 == 0:
                    print(f'  {ok} ok / {fail} fail / {ok+fail} total')
            else:
                fail += 1

    print(f'\nDone: {ok} saved, {fail} failed/invalid → {out}')

    index = []
    for d in sorted(out.iterdir()):
        if d.is_dir() and (d/'notes.chart').exists() and (d/'song.opus').exists():
            meta = {}
            mf = d / 'meta.json'
            if mf.exists():
                meta = json.loads(mf.read_text())
            index.append({'path': str(d), **meta})
    (out / 'index.json').write_text(json.dumps(index, indent=2))
    print(f'Index: {out}/index.json  ({len(index)} entries)')


if __name__ == '__main__':
    main()
