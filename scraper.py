"""
Clone Hero chart scraper — downloads chart+audio pairs from Chorus API.
Usage: python scraper.py --out ./dataset --limit 2000

Targets: chorus.fightthe.pw  (largest public CH chart index with REST API)
Saves each song as:  dataset/<id>/song.opus  +  dataset/<id>/notes.chart
                     dataset/<id>/song.ini   +  dataset/<id>/meta.json
"""

import os, sys, json, time, argparse, zipfile, rarfile, re
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config ────────────────────────────────────────────────────────────────────
CHORUS_API  = 'https://chorus.fightthe.pw/api/search'
CHORUS_SONG = 'https://chorus.fightthe.pw/api/count'
RATE_LIMIT  = 0.4     # seconds between API calls
TIMEOUT     = 30      # seconds per HTTP request
MAX_WORKERS = 4       # parallel downloads
AUDIO_EXTS  = {'.opus', '.ogg', '.mp3', '.wav'}
CHART_NAMES = {'notes.chart', 'notes.mid'}

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_filename(s):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', s)[:80]

def chorus_search(query='', page=0, per_page=50):
    """Single page of Chorus search results."""
    params = {'query': query, 'page': page, 'per_page': per_page,
              'instrument': 'guitar'}
    try:
        r = requests.get(CHORUS_API, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json().get('songs', [])
    except Exception as e:
        print(f'  [warn] Chorus API page {page}: {e}')
        return []

def iter_all_songs(limit=2000):
    """Yield song dicts from Chorus until limit reached."""
    seen, page = set(), 0
    while len(seen) < limit:
        songs = chorus_search(page=page)
        if not songs:
            break
        for s in songs:
            sid = s.get('id') or s.get('link', '')
            if sid and sid not in seen:
                seen.add(sid)
                yield s
                if len(seen) >= limit:
                    return
        page += 1
        time.sleep(RATE_LIMIT)

def find_download_url(song):
    """Extract best direct download URL from a Chorus song dict."""
    # Chorus returns 'directLinks' dict or 'link' string
    dl = song.get('directLinks', {})
    if isinstance(dl, dict):
        # Prefer Archive.org or direct links
        for key in ('archive', 'drive', 'dropbox', 'direct', 'mediafire'):
            if key in dl:
                return dl[key]
        # Any value
        for v in dl.values():
            if v and v.startswith('http'):
                return v
    link = song.get('link', '')
    if link and link.startswith('http'):
        return link
    return None

def download_file(url, dest_path, session):
    """Download url → dest_path. Returns True on success."""
    try:
        with session.get(url, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
        return True
    except Exception as e:
        print(f'  [warn] download {url}: {e}')
        return False

def extract_archive(archive_path, out_dir):
    """Extract zip or rar archive. Returns list of extracted file paths."""
    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(out_dir)
                return [str(out_dir / n) for n in zf.namelist()]
        # Try rar (requires rarfile + unrar binary)
        try:
            with rarfile.RarFile(archive_path) as rf:
                rf.extractall(out_dir)
                return [str(out_dir / n) for n in rf.namelist()]
        except Exception:
            pass
    except Exception as e:
        print(f'  [warn] extract {archive_path.name}: {e}')
    return []

def find_chart_and_audio(extract_dir):
    """Recursively find notes.chart and audio file under extract_dir."""
    chart_path = audio_path = None
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            fl = f.lower()
            fp = Path(root) / f
            if fl in CHART_NAMES and chart_path is None:
                chart_path = fp
            if Path(fl).suffix in AUDIO_EXTS and audio_path is None:
                audio_path = fp
    return chart_path, audio_path

def validate_chart(chart_path):
    """Quick sanity check: chart must have ExpertSingle with at least 50 notes."""
    try:
        text = chart_path.read_text(encoding='utf-8', errors='replace')
        if '[ExpertSingle]' not in text:
            return False
        expert = text.split('[ExpertSingle]')[1].split('}')[0]
        note_count = expert.count(' = N ')
        return note_count >= 50
    except Exception:
        return False

def process_song(song, out_root, session):
    """
    Download + extract + validate one Chorus song.
    Returns path to song folder on success, None on failure.
    """
    song_id   = str(song.get('id', song.get('link', 'unknown')))
    title     = song.get('name', 'unknown')
    artist    = song.get('artist', 'unknown')
    folder    = out_root / safe_filename(f'{song_id}_{artist}_{title}')

    if folder.exists() and (folder / 'notes.chart').exists():
        return str(folder)  # already downloaded

    folder.mkdir(parents=True, exist_ok=True)

    url = find_download_url(song)
    if not url:
        folder.rmdir()
        return None

    # Determine archive extension from URL
    ext = Path(url.split('?')[0]).suffix.lower() or '.zip'
    archive = folder / f'archive{ext}'

    if not download_file(url, archive, session):
        return None

    # Extract
    extract_dir = folder / 'extracted'
    extract_dir.mkdir(exist_ok=True)
    extracted = extract_archive(archive, extract_dir)
    if not extracted:
        return None

    chart_p, audio_p = find_chart_and_audio(extract_dir)
    if chart_p is None or audio_p is None:
        return None

    if not validate_chart(chart_p):
        return None

    # Copy chart + audio to clean location
    import shutil
    shutil.copy(chart_p, folder / 'notes.chart')

    # Convert audio to opus if not already
    target_audio = folder / 'song.opus'
    if audio_p.suffix.lower() == '.opus':
        shutil.copy(audio_p, target_audio)
    else:
        import subprocess
        subprocess.run(
            ['ffmpeg', '-y', '-i', str(audio_p), '-c:a', 'libopus',
             '-b:a', '128k', str(target_audio)],
            capture_output=True)

    if not target_audio.exists():
        return None

    # Save metadata
    meta = {
        'id': song_id, 'title': title, 'artist': artist,
        'genre':  song.get('genre', ''),
        'album':  song.get('album', ''),
        'charter': song.get('charter', ''),
        'source': url,
    }
    (folder / 'meta.json').write_text(json.dumps(meta, indent=2))

    # Cleanup archive + extracted folder to save space
    try:
        archive.unlink()
        import shutil as sh
        sh.rmtree(extract_dir, ignore_errors=True)
    except Exception:
        pass

    return str(folder)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',   default='./dataset', help='Output directory')
    ap.add_argument('--limit', type=int, default=2000, help='Max songs to download')
    ap.add_argument('--workers', type=int, default=MAX_WORKERS)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f'Scraping up to {args.limit} songs → {out}')

    songs    = list(iter_all_songs(args.limit))
    print(f'Found {len(songs)} songs from Chorus API')

    ok = 0; fail = 0
    session = requests.Session()
    session.headers['User-Agent'] = 'CloneHeroTrainer/1.0'

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_song, s, out, session): s for s in songs}
        for future in as_completed(futures):
            result = future.result()
            if result:
                ok += 1
                if ok % 25 == 0:
                    print(f'  Progress: {ok} ok / {fail} fail / {ok+fail} total')
            else:
                fail += 1

    print(f'\nDone: {ok} valid chart pairs saved to {out}')
    print(f'      {fail} failed/invalid')

    # Write index
    index = []
    for d in sorted(out.iterdir()):
        if (d / 'notes.chart').exists() and (d / 'song.opus').exists():
            meta = {}
            if (d / 'meta.json').exists():
                meta = json.loads((d / 'meta.json').read_text())
            index.append({'path': str(d), **meta})
    (out / 'index.json').write_text(json.dumps(index, indent=2))
    print(f'Index written: {out}/index.json  ({len(index)} entries)')


if __name__ == '__main__':
    main()
