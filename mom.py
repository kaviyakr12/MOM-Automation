import gc, os, re, ssl, json, queue, smtplib
import subprocess, tempfile, threading, time
from datetime             import datetime
from pathlib              import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from concurrent.futures   import ThreadPoolExecutor, as_completed 

import numpy as np
import requests
import webview

# ══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════
EMAILS_FILE         = Path.home() / ".mom_emails.json"
SETTINGS_FILE       = Path.home() / ".mom_settings.json"

OLLAMA_URL          = "http://localhost:11434"
OLLAMA_MODEL        = "mistral"
OLLAMA_CTX          = 8192

SAMPLE_RATE         = 16000
CHUNK_FRAMES        = 1024
LIVE_CHUNK_SECS     = 30
OVERLAP_SECS        = 2
FILE_CHUNK_SECS     = 120
MAX_WORKERS         = 4

SILENCE_RMS_THR     = 0.001
LLM_SEND_CHARS      = 12000
MIN_WORDS_PER_CHUNK = 6
MIN_UNIQUE_WORDS    = 50

WHISPER_INITIAL_PROMPT = (
    "This is a formal business meeting. Participants discuss project updates, "
    "timelines, decisions, approvals, and strategy. "
    "Names, product names, dates, and technical terms are common."
)

FILLERS = [
    r'\bum+\b', r'\buh+\b', r'\byou know\b', r'\bbasically\b',
    r'\bi mean\b', r'\bkind of\b', r'\bsort of\b', r'\bokay so\b',
    r'\bactually\b', r'\bliterally\b', r'\bright\b(?=\s)',
    r'\bso\s+so\b', r'\blike\s+like\b',
]


# ══════════════════════════════════════════════════════════════════════
#  AUDIO UTILS
# ══════════════════════════════════════════════════════════════════════
def normalize(audio: np.ndarray) -> np.ndarray:
    return audio / (np.max(np.abs(audio)) + 1e-6)

def is_silent(audio: np.ndarray, threshold: float = SILENCE_RMS_THR) -> bool:
    return float(np.sqrt(np.mean(audio ** 2))) < threshold


# ══════════════════════════════════════════════════════════════════════
#  TRANSCRIPT CLEANER
# ══════════════════════════════════════════════════════════════════════
def clean_transcript(text: str) -> str:
    text = re.sub(r'\[(You|Others|Speaker\s*\d*)\]:\s*', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'([.!?,;])\1+', r'\1', text)
    text = re.sub(r'\b(\w+)(\s+\1){2,}\b', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_fillers(text: str) -> str:
    for f in FILLERS:
        text = re.sub(f, '', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()


# ══════════════════════════════════════════════════════════════════════
#  TRANSCRIPT QUALITY CHECKER
# ══════════════════════════════════════════════════════════════════════
def transcript_quality(text: str) -> dict:
    words        = text.split()
    unique_words = set(w.lower().strip(".,!?;:") for w in words if re.search(r'[a-z]', w, re.I))
    word_count   = len(words)
    unique_count = len(unique_words)
    top_word_freq = (
        max((words.count(w) for w in set(words)), default=0) / word_count
        if word_count > 0 else 1.0
    )
    return {
        "word_count":    word_count,
        "unique_count":  unique_count,
        "top_word_freq": round(top_word_freq, 2),
        "is_good":       (unique_count >= MIN_UNIQUE_WORDS and top_word_freq < 0.4),
    }

def is_junk_chunk(text: str) -> bool:
    words = [w for w in text.split() if re.search(r'[a-zA-Z]', w)]
    return len(words) < MIN_WORDS_PER_CHUNK


# ══════════════════════════════════════════════════════════════════════
#  PERSISTENCE
# ══════════════════════════════════════════════════════════════════════
def load_saved_emails():
    try:    return json.loads(EMAILS_FILE.read_text())
    except: return []

def save_email_ids(ids):
    existing = load_saved_emails()
    EMAILS_FILE.write_text(json.dumps(list(dict.fromkeys(existing + ids)), indent=2))

def load_settings():
    defaults = {
        "sender": "", "password": "",
        "smtp": "smtp.gmail.com", "port": 465,
        "model": "small",
        "mic_device": "", "system_device": "",
        "hf_token": "",
        "ollama_model": OLLAMA_MODEL,
        "summary_sentences": 8,
        "language": "en",
    }
    try:
        saved = json.loads(SETTINGS_FILE.read_text())
        defaults.update(saved)
    except: pass
    return defaults

def save_settings_file(d):
    SETTINGS_FILE.write_text(json.dumps(d, indent=2))

def _st(cb, msg):
    if cb:
        try: cb(str(msg))
        except: pass

def check_ollama() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except:
        return False

def check_ollama_model(model_name: str) -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.status_code != 200: return False
        models = [m["name"] for m in r.json().get("models", [])]
        base   = model_name.split(":")[0].lower()
        return any(base in m.lower() for m in models)
    except:
        return False


# ══════════════════════════════════════════════════════════════════════
#  AUDIO DEVICE DISCOVERY
# ══════════════════════════════════════════════════════════════════════
def list_audio_devices():
    try:
        import sounddevice as sd
        devices = []
        for i, d in enumerate(sd.query_devices()):
            name     = d["name"]
            is_input = d["max_input_channels"] > 0
            is_loop  = any(k in name.lower() for k in [
                "loopback","stereo mix","what u hear","cable output",
                "vb-audio","wave out","virtual audio","wasapi"])
            if is_input or is_loop:
                devices.append({
                    "index": i, "name": name,
                    "inputs": d["max_input_channels"],
                    "samplerate": int(d["default_samplerate"]),
                    "is_loopback": is_loop,
                    "hostapi": sd.query_hostapis(d["hostapi"])["name"],
                })
        return devices
    except Exception as e:
        print(f"[Devices] sounddevice error: {e}")
        return _list_pyaudio_devices()

def _list_pyaudio_devices():
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        devices = []
        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
                if info["maxInputChannels"] > 0:
                    name = info["name"]
                    devices.append({
                        "index": i, "name": name,
                        "inputs": int(info["maxInputChannels"]),
                        "samplerate": int(info["defaultSampleRate"]),
                        "is_loopback": any(k in name.lower() for k in
                            ["stereo mix","loopback","cable","vb-audio"]),
                        "hostapi": "pyaudio",
                    })
            except: pass
        pa.terminate()
        return devices
    except Exception as e:
        print(f"[Devices] pyaudio failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════
#  LIVE RECORDER
# ══════════════════════════════════════════════════════════════════════
class LiveRecorder:
    def __init__(self, q, mic_device=None, system_device=None):
        self.q             = q
        self.mic_device    = mic_device
        self.system_device = system_device
        self.is_recording  = False
        self.is_paused     = False
        self.rms           = 0.0
        self._buf_lock     = threading.Lock()
        self._sd_mic_buf   = []
        self._sd_sys_buf   = []

    def _try_sounddevice(self):
        import sounddevice as sd
        self._sd_mic_buf = []
        self._sd_sys_buf = []

        def mic_cb(indata, frames, time_info, status):
            if not self.is_paused:
                arr = indata[:, 0].copy()
                with self._buf_lock:
                    self._sd_mic_buf.extend(arr.tolist())
                    self.rms = min(1.0, float(np.sqrt(np.mean(arr**2))) / 0.3)

        def sys_cb(indata, frames, time_info, status):
            if not self.is_paused:
                with self._buf_lock:
                    self._sd_sys_buf.extend(indata[:, 0].tolist())

        streams = []
        mic_s = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            callback=mic_cb, blocksize=CHUNK_FRAMES,
            device=self.mic_device or None)
        mic_s.start(); streams.append(mic_s)

        if self.system_device:
            try:
                sys_s = sd.InputStream(
                    device=self.system_device, samplerate=SAMPLE_RATE,
                    channels=1, dtype="float32", callback=sys_cb,
                    blocksize=CHUNK_FRAMES)
                sys_s.start(); streams.append(sys_s)
            except Exception as e:
                print("System audio error:", e)

        overlap_samples = SAMPLE_RATE * OVERLAP_SECS
        chunk_samples   = SAMPLE_RATE * LIVE_CHUNK_SECS
        acc_mic, acc_sys = [], []

        while self.is_recording:
            time.sleep(0.1)
            with self._buf_lock:
                acc_mic.extend(self._sd_mic_buf); self._sd_mic_buf.clear()
                acc_sys.extend(self._sd_sys_buf); self._sd_sys_buf.clear()

            if len(acc_mic) >= chunk_samples:
                mic_arr = normalize(np.array(acc_mic[:chunk_samples], dtype=np.float32))
                sys_arr = None
                if acc_sys:
                    sl = min(len(acc_sys), chunk_samples)
                    sa = np.array(acc_sys[:sl], dtype=np.float32)
                    if len(sa) < chunk_samples:
                        sa = np.pad(sa, (0, chunk_samples - len(sa)))
                    sys_arr = normalize(sa)
                self.q.put({"mic": mic_arr, "system": sys_arr})
                acc_mic = acc_mic[chunk_samples - overlap_samples:]
                acc_sys = acc_sys[chunk_samples - overlap_samples:]

        if acc_mic:
            mic_arr = normalize(np.array(acc_mic, dtype=np.float32))
            sys_arr = normalize(np.array(acc_sys[:len(acc_mic)], dtype=np.float32)) if acc_sys else None
            self.q.put({"mic": mic_arr, "system": sys_arr})

        for s in streams:
            try: s.stop(); s.close()
            except: pass
        self.q.put(None)

    def start(self):
        self.is_recording = True
        self._try_sounddevice()

    def stop(self):   self.is_recording = False
    def pause(self):  self.is_paused    = True
    def resume(self): self.is_paused    = False


# ══════════════════════════════════════════════════════════════════════
#  FFMPEG AUDIO EXTRACTOR
# ══════════════════════════════════════════════════════════════════════
def ffmpeg_to_wav(input_path, on_status=None):
    tmp = tempfile.mktemp(suffix=".wav")
    _st(on_status, "Extracting audio with ffmpeg...")
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", str(input_path), "-vn",
             "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", tmp],
            capture_output=True, timeout=600)
        if proc.returncode == 0 and Path(tmp).exists() and Path(tmp).stat().st_size > 4096:
            _st(on_status, "Audio extracted")
            return tmp, True
        _st(on_status, "ffmpeg produced empty file -- using original")
    except Exception as e:
        _st(on_status, f"ffmpeg error: {e}")
    try:
        if Path(tmp).exists(): os.remove(tmp)
    except: pass
    return str(input_path), False


# ══════════════════════════════════════════════════════════════════════
#  FASTER-WHISPER TRANSCRIBER
# ══════════════════════════════════════════════════════════════════════
class Transcriber:
    def __init__(self):
        self._model_cache = {}

    def _load_model(self, model_name, on_status):
        if model_name in self._model_cache:
            return self._model_cache[model_name]
        _st(on_status, f"Loading Faster-Whisper '{model_name}' (int8)...")
        try:
            from faster_whisper import WhisperModel
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model  = WhisperModel(model_name, device=device, compute_type="int8",
                                  cpu_threads=os.cpu_count())
            self._model_cache[model_name] = model
            _st(on_status, f"Faster-Whisper ready ({device})")
            return model
        except ImportError:
            _st(on_status, "faster-whisper not installed. pip install faster-whisper")
            raise

    def _transcribe_audio(self, model, audio_np, on_status=None, language="en"):
        if is_silent(audio_np):
            return "", []
        try:
            segments, _ = model.transcribe(
                audio_np,
                beam_size=5,
                best_of=5,
                vad_filter=True,
                vad_parameters={"threshold": 0.35, "min_silence_duration_ms": 250},
                condition_on_previous_text=True,
                language=language,
                temperature=0.0,
                no_speech_threshold=0.5,
                log_prob_threshold=-0.8,
                compression_ratio_threshold=2.2,
                word_timestamps=False,
                initial_prompt=WHISPER_INITIAL_PROMPT,
            )
            seg_list, texts = [], []
            for seg in segments:
                if hasattr(seg, 'no_speech_prob') and seg.no_speech_prob > 0.65:
                    continue
                t = seg.text.strip()
                if t:
                    texts.append(t)
                    seg_list.append({"text": t, "start": seg.start, "end": seg.end})

            raw = " ".join(texts).strip()
            if is_junk_chunk(raw):
                return "", []
            return raw, seg_list
        except Exception as e:
            _st(on_status, f"Transcription error: {e}")
            return "", []

    def transcribe_live(self, q, model_name="small", on_status=None,
                        on_sentence=None, language="en"):
        model     = self._load_model(model_name, on_status)
        full_text = []
        n = 0
        while True:
            chunk = q.get()
            if chunk is None:
                break
            n += 1
            _st(on_status, f"Processing live chunk {n}...")
            try:
                mic_text = sys_text = ""
                if chunk.get("mic") is not None:
                    mic_text, _ = self._transcribe_audio(model, chunk["mic"], on_status, language)
                if chunk.get("system") is not None:
                    sys_text, _ = self._transcribe_audio(model, chunk["system"], on_status, language)
                if mic_text:
                    tagged = f"[You]: {mic_text}"
                    full_text.append(tagged)
                    if on_sentence: on_sentence(tagged)
                if sys_text:
                    tagged = f"[Others]: {sys_text}"
                    full_text.append(tagged)
                    if on_sentence: on_sentence(tagged)
            except Exception as e:
                print(f"Error in live chunk {n}: {e}")
            gc.collect()
        return "\n".join(full_text)

    def transcribe_file(self, filepath, model_name="small",
                        on_status=None, language="en"):
        wav_path, is_temp = ffmpeg_to_wav(filepath, on_status)
        try:
            import librosa
            audio, _ = librosa.load(wav_path, sr=16000, mono=True)
        except Exception as e:
            _st(on_status, f"Failed to load audio: {e}")
            if is_temp:
                try: os.remove(wav_path)
                except: pass
            return "", [], []

        chunk_samples = FILE_CHUNK_SECS * SAMPLE_RATE
        total_chunks  = max(1, int(np.ceil(len(audio) / chunk_samples)))
        chunks = []
        for i in range(total_chunks):
            start = i * chunk_samples
            c     = audio[start: start + chunk_samples]
            if len(c) < chunk_samples:
                c = np.pad(c, (0, chunk_samples - len(c)))
            chunks.append(c)

        _st(on_status, f"Split into {total_chunks} x {FILE_CHUNK_SECS}s chunks "
                       f"(workers={MAX_WORKERS})")

        model        = self._load_model(model_name, on_status)
        results      = {}
        silent_count = junk_count = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {
                ex.submit(self._transcribe_audio, model, chunk, on_status, language): i
                for i, chunk in enumerate(chunks)
            }
            done = 0
            for future in as_completed(futures):
                idx  = futures[future]
                done += 1
                try:
                    text, segs = future.result()
                    if text:
                        results[idx] = (text, segs)
                        _st(on_status,
                            f"[{done}/{total_chunks}] chunk {idx+1} "
                            f"({len(text.split())} words)")
                    else:
                        if is_silent(chunks[idx]):
                            silent_count += 1
                        else:
                            junk_count += 1
                except Exception as e:
                    _st(on_status, f"Chunk {idx+1} error: {e}")

        if silent_count: _st(on_status, f"Skipped {silent_count} silent chunk(s)")
        if junk_count:   _st(on_status, f"Filtered {junk_count} junk chunk(s)")

        all_segments, chunk_texts, full_parts = [], [], []
        for idx in sorted(results):
            text, segs = results[idx]
            chunk_texts.append(text)
            full_parts.append(text)
            all_segments.extend(segs)

        full_text = " ".join(full_parts)
        _st(on_status, f"Transcription complete: {len(full_text.split())} words")

        if is_temp:
            try: os.remove(wav_path)
            except: pass

        return full_text, all_segments, chunk_texts

    def process_text(self, text, on_status=None, on_sentence=None):
        from nltk.tokenize import sent_tokenize
        _st(on_status, "Processing pasted transcript...")
        seen, sentences = set(), []
        for s in sent_tokenize(text.strip()):
            n = s.lower().strip()
            if n not in seen and len(n) > 4:
                seen.add(n)
                sentences.append(s.strip())
                if on_sentence: on_sentence(s.strip())
        _st(on_status, f"Processed -- {len(sentences)} sentences")
        return " ".join(sentences)


# ======================================================================
#  SPEAKER DIARIZATION
# ======================================================================
class Diarizer:
    def __init__(self, hf_token=""):
        self.hf_token  = hf_token
        self._pipeline = None

    def _get_pipeline(self, on_status):
        if self._pipeline: return self._pipeline
        try:
            from pyannote.audio import Pipeline
            import torch
            _st(on_status, "Loading speaker diarization model...")
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.0", use_auth_token=self.hf_token)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._pipeline = self._pipeline.to(device)
            _st(on_status, f"Diarization ready ({device})")
            return self._pipeline
        except Exception as e:
            _st(on_status, f"Diarization error: {e}")
            return None

    def diarize(self, audio_path, whisper_segments, on_status=None):
        if not self.hf_token or not self.hf_token.startswith("hf_"):
            _st(on_status, "No HF token -- skipping diarization")
            return [{"speaker": "Unknown", "start": s["start"],
                     "end": s["end"], "text": s["text"]}
                    for s in whisper_segments if s.get("text")]
        pipeline = self._get_pipeline(on_status)
        if not pipeline: return self._fallback(whisper_segments)
        try:
            diarization = pipeline(audio_path)
            turns = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                turns.append({"speaker": speaker, "start": turn.start,
                               "end": turn.end, "text": ""})
            for seg in whisper_segments:
                mid  = (seg["start"] + seg["end"]) / 2
                best = next((t for t in turns if t["start"] <= mid <= t["end"]), None)
                if best: best["text"] += " " + seg["text"]
                elif turns:
                    nearest = min(turns, key=lambda t: abs(t["start"] - mid))
                    nearest["text"] += " " + seg["text"]
            for t in turns:
                t["text"] = t["text"].strip()
                num = int(re.sub(r"\D", "", t["speaker"]) or 0) + 1
                t["speaker"] = f"Speaker {num}"
            return [t for t in turns if t["text"]]
        except Exception as e:
            _st(on_status, f"Diarization failed: {e}")
            return self._fallback(whisper_segments)

    def _fallback(self, segments):
        return [{"speaker": "Unknown", "start": s["start"],
                 "end": s["end"], "text": s["text"]}
                for s in segments if s.get("text")]


# ======================================================================
#  ENTITY EXTRACTOR  -- dates, orgs only (no attendees needed)
# ======================================================================
class Extractor:
    def extract(self, transcript, on_status=None):
        _st(on_status, "Extracting meeting metadata (dates, orgs)...")
        return self._spacy(transcript)

    def _spacy(self, transcript):
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(transcript[:100000])
            dates = list(set(e.text for e in doc.ents if e.label_ == "DATE"))
            orgs  = list(set(e.text for e in doc.ents if e.label_ == "ORG"))
            return {"dates": dates, "orgs": orgs}
        except Exception:
            return {"dates": [], "orgs": []}


def repair_truncated_json(raw: str) -> str:
    s = raw.strip()
    last_clean = max(
        s.rfind('",'),
        s.rfind('"]'),
        s.rfind('"}'),
        s.rfind('",\n'),
    )
    if last_clean > len(s) // 2:
        s = s[:last_clean + 1]

    in_string  = False
    escape     = False
    open_braces   = 0
    open_brackets = 0
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':  open_braces   += 1
        elif ch == '}': open_braces   -= 1
        elif ch == '[': open_brackets += 1
        elif ch == ']': open_brackets -= 1

    if open_brackets > 0:
        s += ']' * open_brackets
    if open_braces > 0:
        s += '}' * open_braces
    return s


# ======================================================================
#  MOM PARSER
# ======================================================================
def parse_mom_json(raw: str):
    text = re.sub(r'```(?:json)?', '', raw).strip()

    def _extract_and_parse(s):
        m = re.search(r'\{.*\}', s, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, TypeError):
            return None

    def _to_list(v):
        if isinstance(v, list):
            return [str(i).strip() for i in v if str(i).strip()]
        if isinstance(v, str):
            return [s.strip() for s in v.split('\n') if s.strip()]
        return []

    def _build(obj):
        return {
            "agenda":     _to_list(obj.get("agenda",     obj.get("topics",         []))),
            "discussion": _to_list(obj.get("discussion", obj.get("key_points",     []))),
            "decisions":  _to_list(obj.get("decisions",  obj.get("decisions_made", []))),
            "next_steps": _to_list(obj.get("next_steps", obj.get("next",           []))),
        }

    obj = _extract_and_parse(text)
    if obj:
        return _build(obj)

    repaired = repair_truncated_json(text)
    obj = _extract_and_parse(repaired)
    if obj:
        return _build(obj)

    return None


def fallback_parse_mom(raw: str) -> dict:
    sections = {"agenda": [], "discussion": [], "decisions": [], "next_steps": []}
    current  = None
    header_map = {
        r'agenda|topic':                   "agenda",
        r'discussion|key.?point|summary':  "discussion",
        r'decision':                       "decisions",
        r'next.?step|follow':              "next_steps",
    }
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        matched = False
        for pattern, key in header_map.items():
            if re.search(pattern, stripped, re.IGNORECASE) and len(stripped) < 80:
                current = key
                matched = True
                break
        if matched:
            continue
        if current and re.match(r'^[-*\u2022\d]', stripped):
            clean = re.sub(r'^[-*\u2022\d]+[.)]\s*', '', stripped).strip()
            if clean and len(clean) > 5:
                sections[current].append(clean)
    return sections


def _strip_leading_names(text: str) -> str:
    """
    Remove leading name references from next-step entries.
    Patterns removed:
      - "John will ..."           -> "Will ..."  (capitalised)
      - "John to ..."             -> "To ..."
      - "John / Sarah to ..."     -> "To ..."
      - "[Name]: ..."             -> "..."
      - "Action: John — ..."      -> "..."
    """
    # Remove bracketed name prefix: [Name]: text
    text = re.sub(r'^\[[^\]]+\]:\s*', '', text).strip()
    # Remove "Name — rest" or "Name: rest" at start (single name, no verb yet)
    text = re.sub(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s*[:\u2014\-]\s*', '', text).strip()
    # Remove "Name will/to/should ..." -> keep "will/to/should ..."
    text = re.sub(
        r'^(?:[A-Z][a-z]+(?:\s*/\s*[A-Z][a-z]+)*)\s+(?=will\b|to\b|should\b|needs?\b|must\b|is\b|are\b)',
        '', text, flags=re.IGNORECASE).strip()
    # Capitalise first letter
    if text:
        text = text[0].upper() + text[1:]
    return text


# ======================================================================
#  SUMMARISER
# ======================================================================
class Summariser:

    def _prepare(self, text: str) -> str:
        return remove_fillers(clean_transcript(text))

    def _ollama_call(self, prompt: str, model: str, on_status,
                     num_predict: int = 2800, label: str = "LLM"):
        read_timeout = max(360, int(num_predict * 1.1))
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model":  model,
                    "stream": True,
                    "prompt": prompt,
                    "options": {
                        "num_predict":    num_predict,
                        "num_ctx":        OLLAMA_CTX,
                        "temperature":    0.0,
                        "seed":           42,
                        "top_p":          1.0,
                        "repeat_penalty": 1.1,
                    },
                },
                timeout=(10, read_timeout),
                stream=True,
            )
            if resp.status_code != 200:
                _st(on_status, f"{label} HTTP {resp.status_code}")
                return None
            parts = []
            for raw_line in resp.iter_lines():
                if not raw_line: continue
                try:
                    obj = json.loads(raw_line)
                    parts.append(obj.get("response", ""))
                    if obj.get("done"): break
                except json.JSONDecodeError:
                    pass
            result = "".join(parts).strip()
            return result if result else None
        except requests.exceptions.Timeout:
            _st(on_status, f"{label} timed out -- extractive fallback")
        except Exception as e:
            _st(on_status, f"{label} error: {e}")
        return None

    def _extractive(self, text: str, n: int, on_status) -> dict:
        try:
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers    import Tokenizer
            from sumy.summarizers.lsa   import LsaSummarizer
            parser    = PlaintextParser.from_string(text, Tokenizer("english"))
            sentences = [str(s) for s in LsaSummarizer()(parser.document, n)]
        except Exception:
            try:
                from nltk.tokenize import sent_tokenize
                sentences = sent_tokenize(text)[:n]
            except Exception:
                sentences = [text[:1500]]
        return {
            "agenda":     [],
            "discussion": sentences,
            "decisions":  [],
            "next_steps": [],
        }

    @staticmethod
    def _build_json_prompt(context: str) -> str:
        return (
            "You are a professional meeting secretary. Your output is a single JSON object.\n\n"
            "BEFORE writing JSON, do these steps mentally:\n"
            "  STEP 1 -- Read the full transcript and identify the major themes.\n"
            "            Group closely related sub-topics into ONE point.\n"
            "            You must produce BETWEEN 5 AND 10 discussion points. Never more than 10.\n"
            "  STEP 2 -- For each point, draft exactly 2 sentences:\n"
            "            Sentence 1 — what topic was raised and what was said about it.\n"
            "            Sentence 2 — the context, outcome, importance, or detail from the transcript.\n"
            "            Each point string = those 2 sentences joined together. No keywords. No fragments.\n"
            "  STEP 3 -- Identify explicit decisions and next steps.\n\n"
            "OUTPUT: Return ONLY valid JSON. No prose. No markdown. No backticks.\n\n"
            "{\n"
            '  "agenda": [\n'
            '    "Brief label for the main subject of this meeting"\n'
            '  ],\n'
            '  "discussion": [\n'
            '    "Two-sentence point: first sentence states the topic and what was discussed. Second sentence gives context, outcome, or key detail from the transcript.",\n'
            '    "... up to 10 points maximum. Consolidate related sub-topics. Never exceed 10."\n'
            '  ],\n'
            '  "decisions": [\n'
            '    "What was decided. Effective date or condition if stated."\n'
            '  ],\n'
            '  "next_steps": [\n'
            '    "Specific follow-up or action item — describe WHAT needs to be done, not WHO will do it. No names."\n'
            '  ]\n'
            "}\n\n"
            "STRICT RULES:\n\n"
            "  DISCUSSION (most important):\n"
            "    - MAXIMUM 10 points. MINIMUM 5 points. Hard limits.\n"
            "    - Every point MUST be exactly 2 full sentences.\n"
            "    - Include specifics: tool names, percentages, dates, product names.\n\n"
            "  NEXT STEPS:\n"
            "    - Describe the ACTION or TASK only. Do NOT mention any person's name.\n"
            "    - BAD: 'John to send the report by Friday'\n"
            "    - GOOD: 'Send the report by Friday'\n"
            "    - BAD: 'Alice will schedule the follow-up meeting'\n"
            "    - GOOD: 'Schedule the follow-up meeting'\n\n"
            "  DECISIONS:\n"
            "    - Only explicit decisions: 'agreed', 'decided', 'approved', 'confirmed'.\n"
            "    - Do NOT mention who decided — describe WHAT was decided.\n"
            "    - If none: [\"No formal decisions recorded\"]\n\n"
            "  GENERAL:\n"
            "    - Only use facts from the transcript. Do not invent anything.\n"
            "    - Do not copy transcript sentences verbatim — paraphrase professionally.\n\n"
            "MEETING TRANSCRIPT:\n"
            "------------------------------------------------------------\n"
            f"{context}\n"
            "------------------------------------------------------------\n"
        )

    def _enforce_discussion_rules(self, sections: dict, on_status) -> dict:
        MAX_POINTS = 10
        MIN_WORDS  = 18

        raw   = sections.get("discussion", [])
        valid = [p for p in raw if len(p.split()) >= MIN_WORDS]
        dropped = len(raw) - len(valid)
        if dropped:
            _st(on_status,
                f"Dropped {dropped} keyword fragment(s) from discussion "
                f"(< {MIN_WORDS} words)")

        if len(valid) > MAX_POINTS:
            _st(on_status,
                f"Capping discussion: {len(valid)} -> {MAX_POINTS} points")
            valid = valid[:MAX_POINTS]

        sections["discussion"] = valid

        # Strip leading names from next_steps
        sections["next_steps"] = [
            _strip_leading_names(s) for s in sections.get("next_steps", [])
        ]

        return sections

    def _run_llm(self, context: str, model: str, on_status,
                 num_predict: int = 2800, label: str = "MOM"):
        if not check_ollama():
            _st(on_status, "Ollama not running -- extractive fallback")
            return None

        _st(on_status, f"Generating MOM with {model} "
                       f"(num_ctx={OLLAMA_CTX}, num_predict={num_predict}, "
                       f"temp=0, seed=42)...")
        raw = self._ollama_call(
            self._build_json_prompt(context),
            model, on_status,
            num_predict=num_predict, label=label)

        if not raw:
            _st(on_status, "LLM returned empty -- extractive fallback")
            return None

        sections = parse_mom_json(raw)
        if sections:
            sections = self._enforce_discussion_rules(sections, on_status)
            _st(on_status,
                f"MOM ready -- "
                f"{len(sections.get('discussion', []))} discussion point(s), "
                f"{len(sections.get('decisions', []))} decision(s)")
            return sections

        _st(on_status, "JSON parse+repair failed -- trying line-based parser")
        sections = fallback_parse_mom(raw)
        if any(sections.values()):
            sections = self._enforce_discussion_rules(sections, on_status)
            _st(on_status,
                f"Line parser: {len(sections.get('discussion', []))} point(s)")
            return sections

        _st(on_status, "Line parser found nothing -- extractive fallback")
        return None

    def summarise(self, transcript: str, speaker_turns=None, n: int = 8,
                  on_status=None, model: str = None) -> dict:
        ollama_model = model or OLLAMA_MODEL
        cleaned      = self._prepare(transcript)

        context = (
            "\n".join(f"[{t['speaker']}]: {t['text']}"
                      for t in speaker_turns if t.get("text"))[:LLM_SEND_CHARS]
            if speaker_turns else cleaned[:LLM_SEND_CHARS]
        )

        if not context.strip():
            return {"agenda": [], "discussion": ["No content found."],
                    "decisions": [], "next_steps": []}

        qc = transcript_quality(cleaned)
        _st(on_status, f"Quality: {qc['unique_count']} unique words, "
                       f"freq={qc['top_word_freq']} -- "
                       f"{'GOOD' if qc['is_good'] else 'LOW'}")
        if not qc["is_good"]:
            _st(on_status, "Low quality transcript -- extractive fallback")
            return self._extractive(cleaned, n, on_status)

        result = self._run_llm(context, ollama_model, on_status,
                               num_predict=2800, label="Live MOM")
        return result if result else self._extractive(cleaned, n, on_status)

    def summarise_from_chunks(self, chunk_texts: list, model: str = None,
                              on_status=None) -> dict:
        if not chunk_texts:
            return {"agenda": [], "discussion": ["No content."],
                    "decisions": [], "next_steps": []}

        ollama_model = model or OLLAMA_MODEL
        merged       = self._prepare(" ".join(chunk_texts))

        if not merged.strip():
            return {"agenda": [], "discussion": ["Transcript empty or all noise."],
                    "decisions": [], "next_steps": []}

        qc = transcript_quality(merged)
        _st(on_status,
            f"Quality: {qc['word_count']} words, {qc['unique_count']} unique, "
            f"freq={qc['top_word_freq']} -- {'GOOD' if qc['is_good'] else 'LOW'}")

        if not qc["is_good"]:
            _st(on_status, "Low quality transcript -- extractive fallback")
            return self._extractive(merged, n=10, on_status=on_status)

        context = merged[:LLM_SEND_CHARS]
        if len(merged) > LLM_SEND_CHARS:
            pct = int(LLM_SEND_CHARS / len(merged) * 100)
            _st(on_status, f"Sending first {pct}% of transcript ({LLM_SEND_CHARS} chars)")
        else:
            _st(on_status, "Full transcript fits in context window")

        result = self._run_llm(context, ollama_model, on_status,
                               num_predict=2800, label="File MOM")
        return result if result else self._extractive(merged, n=10, on_status=on_status)


# ======================================================================
#  MOM BODY BUILDER
#
#  Sections: Meeting Details + Agenda + Discussion + Decisions + Next Steps
#  Removed:  Participants, End-of-Meeting footer
#  Regards:  Meeting Organizer
# ======================================================================
def build_mom_body(mom: dict, sections: dict) -> str:
    SEP1 = "=" * 62
    SEP2 = "=" * 62

    def fmt_list(items: list, fallback: str = "None recorded.") -> str:
        if not items:
            return f"    {fallback}"
        out = []
        for i, item in enumerate(items, 1):
            words   = item.split()
            lines   = []
            prefix  = f"    {i:>2}.  "
            indent  = "         "
            current = prefix
            for word in words:
                if len(current) + len(word) + 1 > 78:
                    lines.append(current.rstrip())
                    current = indent + word
                else:
                    current = (current + " " + word) if current.rstrip() != prefix.rstrip() else current + word
            if current.strip():
                lines.append(current.rstrip())
            out.append("\n".join(lines))
        return "\n\n".join(out)

    agenda     = sections.get("agenda",     [])
    discussion = sections.get("discussion", [])
    decisions  = sections.get("decisions",  [])
    next_steps = sections.get("next_steps", [])

    dates_str = ", ".join(mom["dates"]) if mom.get("dates") else "Not referenced"
    orgs_str  = ", ".join(mom["orgs"])  if mom.get("orgs")  else "Not referenced"

    return (
        f"\n{SEP1}\n"
        f"  MINUTES OF MEETING\n"
        f"{SEP1}\n\n"
        f"  Title          : {mom['title']}\n"
        f"  Date & Time    : {mom['date']}\n"
        f"  Source         : {mom.get('source', 'Recording')}\n"
        f"  Prepared By    : MOM Tool (auto-generated)\n"
        f"  Date References: {dates_str}\n"
        f"  Organisations  : {orgs_str}\n\n"
        f"{SEP2}\n"
        f"  1.  AGENDA / TOPICS COVERED\n"
        f"{SEP2}\n"
        f"{fmt_list(agenda, 'No agenda items identified.')}\n\n"
        f"{SEP2}\n"
        f"  2.  KEY DISCUSSION POINTS\n"
        f"{SEP2}\n"
        f"{fmt_list(discussion, 'No discussion points recorded.')}\n\n"
        f"{SEP2}\n"
        f"  3.  DECISIONS MADE\n"
        f"{SEP2}\n"
        f"{fmt_list(decisions, 'No formal decisions recorded.')}\n\n"
        f"{SEP2}\n"
        f"  4.  NEXT STEPS\n"
        f"{SEP2}\n"
        f"{fmt_list(next_steps, 'No next steps mentioned.')}\n\n"
        f"Regards,\n"
        f"Meeting Organizer\n"
    )


# ======================================================================
#  PYWEBVIEW API
# ======================================================================
class API:
    def __init__(self):
        self.recorder        = None
        self.rec_thread      = None
        self.audio_q         = queue.Queue()
        self.upload_path     = None
        self.window          = None
        self.live_transcript = ""
        self.transcriber     = Transcriber()
        self.extractor       = Extractor()
        self.summariser      = Summariser()
        self._diarizer       = None

    def _diarizer_for(self, hf_token):
        if self._diarizer is None or self._diarizer.hf_token != hf_token:
            self._diarizer = Diarizer(hf_token)
        return self._diarizer

    def _jlog(self, msg, cls="ok"):
        if self.window:
            try:
                self.window.evaluate_js(f'window.log({json.dumps(str(msg))},"{cls}")')
            except: pass

    def _jsentence(self, s):
        if self.window:
            try:
                self.window.evaluate_js(f'window.appendTranscript({json.dumps(s)})')
            except: pass

    def get_audio_devices(self):
        try:    return list_audio_devices()
        except Exception as e:
            return [{"index": -1, "name": f"Error: {e}", "inputs": 0,
                     "is_loopback": False, "samplerate": 44100, "hostapi": ""}]

    def start_recording(self, params=None):
        if isinstance(params, str):
            try:   params = json.loads(params)
            except: params = {}
        if params is None: params = {}
        settings      = load_settings()
        mic_device    = params.get("mic_device")    or settings.get("mic_device")    or None
        system_device = params.get("system_device") or settings.get("system_device") or None
        language      = settings.get("language", "en")

        self.audio_q    = queue.Queue()
        self.recorder   = LiveRecorder(self.audio_q, mic_device=mic_device,
                                       system_device=system_device)
        self.rec_thread = threading.Thread(target=self.recorder.start, daemon=True)
        self.rec_thread.start()

        def run_transcription():
            try:
                text = self.transcriber.transcribe_live(
                    self.audio_q,
                    model_name=settings.get("model", "small"),
                    on_status=self._jlog,
                    on_sentence=self._jsentence,
                    language=language)
                self.live_transcript = text
            except Exception as e:
                self._jlog(f"Live transcription error: {e}", "err")

        threading.Thread(target=run_transcription, daemon=True).start()
        src = "Mic only" if not system_device else f"Mic + System ({system_device[:30]})"
        self._jlog(f"Recording started -- {src} | lang={language}")
        return {"ok": True, "source": src}

    def stop_recording(self):
        if self.recorder: self.recorder.stop()
        return "ok"

    def pause_recording(self):
        if self.recorder: self.recorder.pause()
        return "ok"

    def resume_recording(self):
        if self.recorder: self.recorder.resume()
        return "ok"

    def get_vu_level(self):
        return round(self.recorder.rms, 3) if self.recorder else 0.0

    def upload_file(self):
        result = self.window.create_file_dialog(
            webview.OPEN_DIALOG, allow_multiple=False,
            file_types=(
                'Audio and Video (*.mp3;*.mp4;*.wav;*.m4a;*.ogg;*.flac;*.mkv;*.avi;*.mov;*.webm)',
                'All files (*.*)'
            ))
        if result and len(result) > 0:
            self.upload_path = result[0]
            return {"path": self.upload_path, "name": Path(self.upload_path).name}
        return None

    def generate_mom_from_file(self, params=None):
        if isinstance(params, str):
            try:   params = json.loads(params)
            except: params = {}
        if params is None: params = {}

        title             = params.get("title", "Meeting")
        model_name        = params.get("model", "small")
        do_diar           = params.get("enable_diarization", False)
        use_chunk_summary = params.get("use_chunk_summary", True)
        settings          = load_settings()
        hf_token          = settings.get("hf_token", "")
        language          = settings.get("language", "en")
        ollama_model      = settings.get("ollama_model", OLLAMA_MODEL)

        if not self.upload_path:
            return {"error": "No file selected. Please use Browse Files first."}

        filepath          = self.upload_path
        source_label      = f"File: {Path(filepath).name}"
        self.upload_path  = None

        if not check_ollama():
            self._jlog("Ollama not running -- extractive summary will be used", "warn")
        elif not check_ollama_model(ollama_model):
            self._jlog(f"Model '{ollama_model}' not pulled. "
                       f"Run: ollama pull {ollama_model}", "warn")

        t0 = time.time()
        try:
            self._jlog(f"Processing: {Path(filepath).name} | "
                       f"whisper={model_name} | lang={language} | llm={ollama_model}")

            full_transcript, whisper_segments, chunk_texts = \
                self.transcriber.transcribe_file(
                    filepath, model_name=model_name,
                    on_status=self._jlog, language=language)

            if not full_transcript.strip():
                return {"error": "No speech detected. "
                        "Try a larger Whisper model (medium/large) or better audio."}

            self._jlog(f"Transcript: {len(full_transcript.split())} words")

            speaker_turns = None
            if do_diar and whisper_segments and hf_token:
                self._jlog("Running speaker diarization...")
                wav_path, is_temp = ffmpeg_to_wav(filepath, self._jlog)
                try:
                    speaker_turns = self._diarizer_for(hf_token).diarize(
                        wav_path, whisper_segments, on_status=self._jlog)
                    if speaker_turns:
                        speakers = set(t["speaker"] for t in speaker_turns)
                        self._jlog(f"Speakers: {', '.join(speakers)}")
                finally:
                    if is_temp and Path(wav_path).exists():
                        os.remove(wav_path)

            self._jlog("Extracting meeting metadata...")
            analysis = self.extractor.extract(
                clean_transcript(full_transcript), on_status=self._jlog)

            if use_chunk_summary and chunk_texts:
                self._jlog("Generating structured MOM...")
                sections = self.summariser.summarise_from_chunks(
                    chunk_texts, model=ollama_model, on_status=self._jlog)
            else:
                sections = self.summariser.summarise(
                    full_transcript, speaker_turns=speaker_turns,
                    n=int(settings.get("summary_sentences", 8)),
                    on_status=self._jlog, model=ollama_model)

            full_transcript = None
            chunk_texts     = None
            gc.collect()

            mom = {
                "title":  title,
                "date":   datetime.now().strftime("%d %B %Y, %H:%M"),
                "source": source_label,
                "dates":  analysis["dates"],
                "orgs":   analysis["orgs"],
            }
            body       = build_mom_body(mom, sections)
            total_time = time.time() - t0
            self._jlog(f"MOM generated in {total_time:.1f}s!")
            return {
                "body":    body,
                "dates":   mom["dates"],
                "orgs":    mom["orgs"],
                "elapsed": round(total_time, 1),
            }
        except Exception as e:
            import traceback; traceback.print_exc()
            return {"error": str(e)}

    def generate_mom_from_recording(self, params=None):
        if isinstance(params, str):
            try:   params = json.loads(params)
            except: params = {}
        if params is None: params = {}
        title      = params.get("title", "Meeting")
        model_name = params.get("model", "small")
        settings   = load_settings()
        language   = settings.get("language", "en")
        t0 = time.time()
        try:
            self._jlog("Transcribing live recording...")
            transcript = self.transcriber.transcribe_live(
                self.audio_q, model_name=model_name,
                on_status=self._jlog, on_sentence=self._jsentence,
                language=language)
            if not transcript.strip():
                return {"error": "No speech detected."}
            self._jlog(f"Transcript: {len(transcript.split())} words")

            analysis = self.extractor.extract(
                clean_transcript(transcript), on_status=self._jlog)
            sections = self.summariser.summarise(
                transcript,
                n=int(settings.get("summary_sentences", 8)),
                on_status=self._jlog,
                model=settings.get("ollama_model", OLLAMA_MODEL))

            transcript = None
            gc.collect()

            mom = {
                "title":  title,
                "date":   datetime.now().strftime("%d %B %Y, %H:%M"),
                "source": "Live Recording",
                "dates":  analysis["dates"],
                "orgs":   analysis["orgs"],
            }
            body = build_mom_body(mom, sections)
            self._jlog(f"MOM generated in {time.time()-t0:.1f}s!")
            return {"body": body, "dates": mom["dates"], "orgs": mom["orgs"]}
        except Exception as e:
            return {"error": str(e)}

    def generate_mom_from_text(self, params=None):
        if isinstance(params, str):
            try:   params = json.loads(params)
            except: params = {}
        if params is None: params = {}
        title    = params.get("title", "Meeting")
        raw_text = params.get("text", "")
        settings = load_settings()
        t0 = time.time()
        try:
            transcript = self.transcriber.process_text(raw_text, self._jlog, self._jsentence)
            if not transcript.strip():
                return {"error": "No text provided."}

            analysis = self.extractor.extract(
                clean_transcript(transcript), on_status=self._jlog)
            sections = self.summariser.summarise(
                transcript,
                n=int(settings.get("summary_sentences", 8)),
                on_status=self._jlog,
                model=settings.get("ollama_model", OLLAMA_MODEL))

            transcript = None
            gc.collect()

            mom = {
                "title":  title,
                "date":   datetime.now().strftime("%d %B %Y, %H:%M"),
                "source": "Pasted Transcript",
                "dates":  analysis["dates"],
                "orgs":   analysis["orgs"],
            }
            body = build_mom_body(mom, sections)
            self._jlog(f"MOM generated in {time.time()-t0:.1f}s!")
            return {"body": body, "dates": mom["dates"], "orgs": mom["orgs"]}
        except Exception as e:
            return {"error": str(e)}

    def save_settings(self, data):
        save_settings_file(data); return "ok"

    def load_settings(self):
        return load_settings()


# ======================================================================
#  ENTRY POINT
# ======================================================================
if __name__ == '__main__':
    print("=" * 68)
    print("  MEETING MOM TOOL  v9  -- Streamlined MOM Edition")
    print(f"  LLM    : {OLLAMA_MODEL}  |  num_ctx={OLLAMA_CTX}  |  temp=0")
    print(f"  Output : Meeting Details + Agenda + Discussion + Decisions + Next Steps")
    print(f"  Whisper: beam=5, best_of=5, condition_on_prev=True")
    print("=" * 68)

    if check_ollama():
        if check_ollama_model(OLLAMA_MODEL):
            print(f"Ollama        --  running | '{OLLAMA_MODEL}' ready")
        else:
            print(f"Ollama        --  running BUT '{OLLAMA_MODEL}' not pulled")
            print(f"                  Run: ollama pull {OLLAMA_MODEL}")
    else:
        print("Ollama        --  NOT running")
        print("                  Run: ollama serve")
        print(f"                  Then: ollama pull {OLLAMA_MODEL}")

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        print("ffmpeg        --  available")
    except FileNotFoundError:
        print("ffmpeg        --  NOT FOUND (brew install ffmpeg / choco install ffmpeg)")

    try:
        import sounddevice; print("sounddevice   --  available")
    except ImportError:
        print("sounddevice   -- not installed (pip install sounddevice)")

    print("\n-- Input Devices -------------------------------------------------")
    try:
        for d in list_audio_devices():
            tag = " < LOOPBACK" if d["is_loopback"] else ""
            print(f"  [{d['index']:>2}] {d['name'][:52]}{tag}")
    except Exception as e:
        print(f"  Could not list devices: {e}")
    print("-" * 68)

    api    = API()
    window = webview.create_window(
        title="Meeting MOM Tool v9",
        url=f"file://{os.path.abspath('mom.html')}",
        js_api=api,
        width=1200, height=860,
        min_size=(900, 640), resizable=True)
    api.window = window
    webview.start(gui='edgechromium')