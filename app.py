import re
import os
import time
import datetime
import math
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sentence_transformers import SentenceTransformer, util
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import yt_dlp
import streamlit as st
from dotenv import load_dotenv

# Set page config as the first Streamlit command
st.set_page_config(page_title="Semantic YouTube Search", layout="wide", initial_sidebar_state="collapsed")

# --- Model & Data Loading (Cached for performance) ---
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data packages."""
    for package in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
        nltk.download(package, quiet=True)
    return True

@st.cache_resource
def load_model():
    """Load the Sentence Transformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

with st.spinner("Initializing AI models and resources..."):
    download_nltk_data()
    model = load_model()

# --- Configuration ---
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
COOKIES_FILE = "cookies.txt"

# --- Core Logic Functions ---
def get_authenticated_service(api_key):
    """Authenticate and return YouTube API service."""
    if not api_key:
        st.error("YouTube API key not found. Please create a .env file with your YOUTUBE_API_KEY.")
        return None
    try:
        return build("youtube", "v3", developerKey=api_key)
    except Exception as e:
        st.error(f"Failed to initialize YouTube API: {e}")
        return None

def clean_text(text):
    """Clean and preprocess text for analysis."""
    if not text: return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

@st.cache_data(ttl=3600)
def get_video_captions(_video_id, _cookies_file, timed=False):
    """Fetch video captions using yt-dlp. Caches results for 1 hour."""
    ydl_opts = {
        'skip_download': True, 'writeautomaticsub': True,
        'subtitleslangs': ['en'], 'subtitlesformat': 'vtt',
        'outtmpl': f'{_video_id}.%(ext)s', 'quiet': True, 'no_warnings': True,
    }
    if _cookies_file and os.path.exists(_cookies_file):
        ydl_opts['cookiefile'] = _cookies_file

    vtt_file = f"{_video_id}.en.vtt"
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={_video_id}"])
        
        if not os.path.exists(vtt_file): return [] if timed else ""
        
        with open(vtt_file, "r", encoding="utf-8") as f: lines = f.readlines()
        
        captions, full_text = [], []
        for i, line in enumerate(lines):
            if "-->" in line:
                try:
                    timestamp = line.split(" --> ")[0]
                    h, m, s_ms = timestamp.split(':')
                    s, ms = s_ms.split('.')
                    start_seconds = int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000
                    caption_text = re.sub(r'<[^>]+>', '', lines[i+1].strip())
                    captions.append((caption_text, start_seconds))
                    full_text.append(caption_text)
                except (ValueError, IndexError): continue
        
        if timed: return captions
        else: return clean_text(" ".join(full_text))

    finally:
        if os.path.exists(vtt_file): os.remove(vtt_file)

@st.cache_data(ttl=3600)
def search_videos(_youtube, query, max_results=15):
    """Search YouTube for a query and return video details."""
    try:
        search_response = _youtube.search().list(
            q=query, part="id,snippet", maxResults=max_results, type="video", order="relevance"
        ).execute()
        
        video_ids = [item["id"]["videoId"] for item in search_response.get("items", [])]
        if not video_ids: return []

        video_response = _youtube.videos().list(part="snippet,statistics", id=",".join(video_ids)).execute()
        
        return [{
            "video_id": item["id"], "title": item["snippet"]["title"],
            "description": item["snippet"]["description"],
            "likes": int(item["statistics"].get("likeCount", 0)),
            "views": int(item["statistics"].get("viewCount", 0)),
        } for item in video_response.get("items", [])]
    except HttpError as e:
        if "quota" in str(e).lower(): st.error("YouTube API quota exceeded.")
        else: st.error(f"An API error occurred: {e}")
        return []

def analyze_videos(_youtube, query, videos):
    """
    Re-ranks videos based on a composite score of semantic relevance and quality metrics.
    """
    cleaned_query = clean_text(query)
    query_embedding = model.encode(cleaned_query)
    
    video_data = []
    max_views = max((v['views'] for v in videos), default=1)
    
    for video in videos:
        # 1. Semantic Relevance Score (NEW: weighted average)
        title_embedding = model.encode(clean_text(video['title']))
        desc_embedding = model.encode(clean_text(video['description']))
        captions_text = get_video_captions(video["video_id"], COOKIES_FILE)
        captions_embedding = model.encode(captions_text) if captions_text else title_embedding

        relevance_score = (
            util.cos_sim(query_embedding, title_embedding).item() * 0.50 +
            util.cos_sim(query_embedding, desc_embedding).item() * 0.20 +
            util.cos_sim(query_embedding, captions_embedding).item() * 0.30
        )
        
        # 2. Quality Score
        view_score = math.log10(video['views'] + 1) / math.log10(max_views + 1)
        like_ratio = video['likes'] / (video['views'] + 1)
        quality_score = (view_score * 0.4) + (like_ratio * 0.6)

        # 3. Composite Score
        composite_score = (relevance_score * 0.75) + (quality_score * 0.25)
        
        video_data.append({**video, "relevance_score": relevance_score * 100, "composite_score": composite_score})

    return sorted(video_data, key=lambda x: x["composite_score"], reverse=True)

@st.cache_data(ttl=3600)
def find_top_shorts(query, video_id, _cookies_file, num_shorts=3, duration=60, step=30):
    """Finds the top N non-overlapping relevant video segments."""
    timed_captions = get_video_captions(video_id, _cookies_file, timed=True)
    if not timed_captions or len(timed_captions) < 5: return []

    query_embedding = model.encode(clean_text(query))
    
    segments = []
    video_duration = timed_captions[-1][1]
    
    for start_time in range(0, int(video_duration), step):
        end_time = start_time + duration
        segment_text = " ".join([text for text, ts in timed_captions if start_time <= ts < end_time])
        if len(segment_text.split()) > 10:
            segments.append({"text": segment_text, "start": start_time, "end": end_time})

    if not segments: return []
        
    segment_embeddings = model.encode([clean_text(s["text"]) for s in segments])
    scores = util.cos_sim(query_embedding, segment_embeddings)[0]
    
    top_shorts = []
    used_indices = set()
    
    # Get sorted indices of scores in descending order
    sorted_score_indices = scores.argsort(descending=True)
    
    for idx in sorted_score_indices:
        if idx in used_indices:
            continue
        
        current_short = segments[idx]
        current_short['score'] = scores[idx].item()
        top_shorts.append(current_short)
        
        # Mark overlapping segments as used
        for i, segment in enumerate(segments):
            # Check for time overlap
            if max(current_short['start'], segment['start']) < min(current_short['end'], segment['end']):
                used_indices.add(i)
                
        if len(top_shorts) >= num_shorts:
            break
            
    return top_shorts

def download_video_segment(video_id, start_time, end_time):
    """Download a specific time segment of a video."""
    output_filename = f"short_{video_id}_{int(start_time)}.mp4"
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_filename,
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegVideoRemuxer',
            'preferedformat': 'mp4',
        }],
        'download_ranges': yt_dlp.utils.download_range_func(None, [(start_time, end_time)]),
        # This option can make cutting much faster
        'force_keyframes_at_cuts': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        return output_filename
    except Exception as e:
        st.error(f"Failed to download clip: {e}")
        return None

# --- Streamlit UI ---
def main():
    st.markdown("""
        <style>
            .main { background-color: #0E1117; }
            .stTextInput>div>div>input { background-color: #1a1e29; }
            .stButton>button { border-radius: 8px; background-color: #4A4AFF; color: white; }
            .video-card {
                background-color: #161B22; border-radius: 12px; padding: 20px;
                margin-bottom: 20px; border: 1px solid #30363D;
            }
            .video-title { font-size: 1.3em; font-weight: 600; color: #E6EDF3; }
            .short-card {
                background-color: #0D1117; border-radius: 8px; padding: 15px;
                margin-top: 15px; border: 1px solid #30363D;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎬 Semantic YouTube Search")
    st.markdown("Go beyond keywords. Find videos based on the *meaning* of their content.")

    query = st.text_input("What are you looking for?", placeholder="e.g., the physics of interstellar travel", key="query_input")
    
    if st.button("🔍 Search", use_container_width=True):
        st.session_state.relevant_videos = None # Clear previous results
        if not query.strip():
            st.error("Please enter a search query.")
            return
        
        youtube = get_authenticated_service(YOUTUBE_API_KEY)
        if not youtube: return

        with st.spinner("Searching and re-ranking videos for relevance..."):
            initial_videos = search_videos(youtube, query)
            if not initial_videos:
                st.warning("No videos found. Try a different query.")
                return
            st.session_state.relevant_videos = analyze_videos(youtube, query, initial_videos)
            st.session_state.query = query

    if 'relevant_videos' in st.session_state and st.session_state.relevant_videos:
        st.markdown("---")
        st.subheader(f"Top 5 Most Relevant Videos for: \"{st.session_state.query}\"")
        
        for i, video in enumerate(st.session_state.relevant_videos[:5]):
            with st.container(border=True): # Use a container for each video card
                st.markdown(f"<p class='video-title'>{video['title']}</p>", unsafe_allow_html=True)
                
                col_img, col_info = st.columns([1, 2])
                with col_img:
                    st.image(f"https://img.youtube.com/vi/{video['video_id']}/mqdefault.jpg")
                    st.markdown(f"<a href='https://www.youtube.com/watch?v={video['video_id']}' target='_blank'>Watch Full Video →</a>", unsafe_allow_html=True)

                with col_info:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Relevance", f"{video['relevance_score']:.1f}%")
                    m2.metric("Views", f"{video['views']:,}")
                    m3.metric("Likes", f"{video['likes']:,}")

                if st.button("💡 Find Top 3 Clips", key=f"find_{video['video_id']}", help="Analyze transcript to find the 3 most relevant clips"):
                    with st.spinner("Analyzing transcript for best clips..."):
                        top_shorts = find_top_shorts(st.session_state.query, video['video_id'], COOKIES_FILE)
                        st.session_state[f"shorts_{video['video_id']}"] = top_shorts
                
                shorts_key = f"shorts_{video['video_id']}"
                if shorts_key in st.session_state:
                    shorts_data = st.session_state[shorts_key]
                    if shorts_data:
                        st.markdown("##### Top 3 Relevant Clips:")
                        cols = st.columns(len(shorts_data))
                        for idx, short in enumerate(shorts_data):
                            with cols[idx]:
                                st.markdown(f"<div class='short-card'>", unsafe_allow_html=True)
                                start_time = int(short['start'])
                                st.video(f"https://www.youtube.com/watch?v={video['video_id']}", start_time=start_time)
                                
                                with st.expander("Show Transcript & Download"):
                                    st.caption(f"Clip starts at {datetime.timedelta(seconds=start_time)}")
                                    st.info(f"\"{short['text'][:200]}...\"")
                                    if st.button("💾 Download Clip", key=f"dl_{video['video_id']}_{idx}"):
                                        with st.spinner("Downloading clip..."):
                                            filepath = download_video_segment(video['video_id'], start_time, int(short['end']))
                                            if filepath and os.path.exists(filepath):
                                                with open(filepath, "rb") as f:
                                                    st.download_button("Click to Download MP4", f.read(), file_name=os.path.basename(filepath), mime="video/mp4")
                                st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.warning("Could not find any clips in this video that are highly relevant to your query.")

if __name__ == "__main__":
    main()
