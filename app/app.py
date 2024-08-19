import streamlit as st
import requests
from googleapiclient.discovery import build
from pytubefix import YouTube
from pydub import AudioSegment
import os
import stat
from transformers import pipeline
from datetime import datetime
from openai import OpenAI
import json
from huggingface_hub import InferenceClient
from bs4 import BeautifulSoup
import re

# Hugging Face 토큰 파일 경로
huggingface_token_path = '/Users/hyemin/Downloads/token'

# Hugging Face 토큰을 읽어 환경 변수로 설정
def set_huggingface_token(token_path):
    with open(token_path, 'r') as file:
        token = file.read().strip()
        os.environ['HF_HUB_TOKEN'] = token
        return token

# Hugging Face 토큰 설정
huggingface_token = set_huggingface_token(huggingface_token_path)

# API 키를 안전하게 설정
youtube_api_key = st.secrets["YOUTUBE_API_KEY"]
openai_api_key = st.secrets["OPENAI_API_KEY"]
google_api_key = st.secrets["GOOGLE_API_KEY"]
google_cse_id = st.secrets["GOOGLE_CSE_ID"]

# Whisper 모델 로드
whisper_model = pipeline("automatic-speech-recognition", model="openai/whisper-base")

# 유튜브 API 클라이언트 초기화
youtube = build('youtube', 'v3', developerKey=youtube_api_key)

def search_youtube(query, max_results=5):
    try:
        request = youtube.search().list(q=query, part='snippet', type='video', maxResults=max_results)
        response = request.execute()
        video_urls = [f"https://www.youtube.com/watch?v={item['id']['videoId']}" for item in response['items']]
        return video_urls
    except Exception as e:
        print(f"Error searching YouTube: {e}")
        return []

def download_subtitles(video_url):
    try:
        yt = YouTube(video_url)
        caption = yt.captions.get('en')
        return caption.generate_srt_captions() if caption else None
    except Exception as e:
        print(f"Error downloading subtitles: {e}")
        return None

def download_audio(video_url, index):
    try:
        yt = YouTube(video_url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        audio_file = audio_stream.download(filename=f'audio_{index}.mp4')
        return audio_file
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None

def convert_audio_to_wav(input_file, index):
    try:
        output_file = f"audio_{index}.wav"
        audio = AudioSegment.from_file(input_file)
        audio.export(output_file, format="wav")
        os.chmod(output_file, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH)
        return output_file
    except Exception as e:
        print(f"Error converting audio: {e}")
        return None

def transcribe_audio_whisper(audio_file):
    try:
        return whisper_model(audio_file)["text"]
    except Exception as e:
        print(f"Error in audio transcription: {e}")
        return None

def generate_conversation(text, model="gpt-4o"):
    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful agent."},
                {"role": "user", "content": f"Create a conversation script from this text: {text}"}
            ],
            max_tokens=4008
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating conversation: {e}")
        return "Error in generating conversation."

def generate_llama_conversation(prompt, token, max_length=2048):
    try:
        API_URL = f"https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
        headers = {"Authorization": f"Bearer {token}"}

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": max_length
            }
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()

        generated_text = response.json()
        if isinstance(generated_text, list):
            return generated_text[0].get('generated_text', 'No text generated.')
        else:
            print("Unexpected response format: ", generated_text)
            return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        print("Response content: ", e.response.content)
        return None
    except KeyError:
        print("Key error: response JSON structure may have changed.")
        return None

def split_text(text, max_length=4008):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def process_conversation_with_llama3(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        conversation = f.read()

    corrected_conversation = generate_llama_conversation(conversation, huggingface_token)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(corrected_conversation)
    print(f"Llama3가 처리한 대화가 {output_file}에 저장되었습니다.")

def process_conversation_with_chatgpt(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        conversation = f.read()

    try:
        corrected_conversation = generate_conversation(f"Edit and improve this conversation:\n\n{conversation}")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(corrected_conversation)
        print(f"ChatGPT가 처리한 대화가 {output_file}에 저장되었습니다.")
    except Exception as e:
        print(f"Error processing conversation with ChatGPT: {e}")

# Google 검색 결과 크롤링
def google_search(query, api_key, cse_id, **kwargs):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {'key': api_key, 'cx': cse_id, 'q': query}
    params.update(kwargs)
    response = requests.get(url, params=params)
    search_items = response.json().get('items', [])
    return {'items': [{'title': item['title'], 'link': item['link']} for item in search_items]}

def fetch_page_content(url):
    try:
        response = requests.get(url)
        return BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        print(f"Failed to fetch {url}: {str(e)}")
        return None

def find_dialogues(soup):
    pattern = re.compile(r"(Officer|Agent|Customs|관광객):\s*(.+)")
    text = soup.get_text()
    return pattern.findall(text)

def scrape_search_results(results):
    dialogues_found = []
    for item in results['items']:
        soup = fetch_page_content(item['link'])
        if soup:
            dialogues = find_dialogues(soup)
            if dialogues:
                dialogues_found.append((item['title'], item['link'], dialogues))
    return dialogues_found

# Streamlit 앱 정의
def main():
    st.title("YouTube Video and Google Search Conversation Generator")

    query = st.text_input("Enter a search query", "immigration conversation at airport")

    if st.button("Extract YouTube and Google Text"):
        if query:
            video_urls = search_youtube(query)
            all_texts = []

            # 타임스탬프 폴더 생성
            timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(timestamp_folder, exist_ok=True)
            st.session_state['folder_name'] = timestamp_folder  # 폴더 이름을 세션 상태에 저장

            # 유튜브 텍스트 추출
            youtube_texts = []
            for index, url in enumerate(video_urls):
                st.write(f"Processing video {index+1}/{len(video_urls)}: {url}")
                subtitles = download_subtitles(url)
                if subtitles:
                    youtube_texts.append(subtitles)
                else:
                    st.write(f"No subtitles found for video: {url}, attempting to transcribe audio.")
                    audio_file = download_audio(url, index)
                    if audio_file:
                        wav_file = convert_audio_to_wav(audio_file, index)
                        if wav_file:
                            transcription = transcribe_audio_whisper(wav_file)
                            if transcription:
                                youtube_texts.append(transcription)
                                st.write(f"Transcription for video {url}: {transcription[:100]}...")

            combined_youtube_text = "\n\n".join(youtube_texts)
            youtube_filename = os.path.join(timestamp_folder, "YouTube_Transcription.txt")
            with open(youtube_filename, "w", encoding="utf-8") as f:
                f.write(combined_youtube_text)
            st.write(f"Transcriptions saved to {youtube_filename}")

            # Google 검색 결과 크롤링 및 저장
            google_results = google_search(query, google_api_key, google_cse_id)
            found_dialogues = scrape_search_results(google_results)
            google_texts = []
            for title, url, dialogues in found_dialogues:
                google_texts.append(f"Title: {title}\nLink: {url}\nDialogues:\n")
                for speaker, line in dialogues:
                    google_texts.append(f"{speaker}: {line}\n")
                google_texts.append("\n")
            
            combined_google_text = "\n".join(google_texts)
            google_filename = os.path.join(timestamp_folder, "Google_Search_Results.txt")
            with open(google_filename, "w", encoding="utf-8") as f:
                f.write(combined_google_text)
            st.write(f"Google search results saved to {google_filename}")

    if st.button("Generate GPT Conversations"):
        # 기존 폴더에서 대화 생성 및 파일 합본
        folder_name = st.session_state.get('folder_name')
        
        if folder_name:
            youtube_filename = os.path.join(folder_name, "YouTube_Transcription.txt")
            google_filename = os.path.join(folder_name, "Google_Search_Results.txt")
            
            combined_gpt_filename = os.path.join(folder_name, "Combined_GPT_Conversations.txt")
            with open(combined_gpt_filename, "w", encoding="utf-8") as outfile:
                if os.path.exists(youtube_filename):
                    with open(youtube_filename, "r", encoding="utf-8") as f:
                        youtube_text = f.read().strip()
                    gpt_youtube_conversation = generate_conversation(youtube_text)
                    gpt_youtube_filename = os.path.join(folder_name, "chatgpt_youtube_conversation.txt")
                    with open(gpt_youtube_filename, "w", encoding="utf-8") as f:
                        f.write(gpt_youtube_conversation)
                    st.write(f"GPT YouTube Conversation saved to {gpt_youtube_filename}")
                    outfile.write("### GPT YouTube Conversation ###\n")
                    outfile.write(gpt_youtube_conversation)
                    outfile.write("\n\n")
                
                if os.path.exists(google_filename):
                    with open(google_filename, "r", encoding="utf-8") as f:
                        google_text = f.read().strip()
                    gpt_google_conversation = generate_conversation(google_text)
                    gpt_google_filename = os.path.join(folder_name, "chatgpt_google_conversation.txt")
                    with open(gpt_google_filename, "w", encoding="utf-8") as f:
                        f.write(gpt_google_conversation)
                    st.write(f"GPT Google Conversation saved to {gpt_google_filename}")
                    outfile.write("### GPT Google Conversation ###\n")
                    outfile.write(gpt_google_conversation)
                    outfile.write("\n\n")
                
            st.write(f"Combined GPT conversations saved to {combined_gpt_filename}")
        else:
            st.write("Please extract YouTube and Google text first to create the necessary files.")

    if st.button("Generate Llama Conversations"):
        # 기존 폴더에서 대화 생성 및 파일 합본
        folder_name = st.session_state.get('folder_name')
        
        if folder_name:
            youtube_filename = os.path.join(folder_name, "YouTube_Transcription.txt")
            google_filename = os.path.join(folder_name, "Google_Search_Results.txt")
            
            combined_llama_filename = os.path.join(folder_name, "Combined_Llama_Conversations.txt")
            with open(combined_llama_filename, "w", encoding="utf-8") as outfile:
                if os.path.exists(youtube_filename):
                    with open(youtube_filename, "r", encoding="utf-8") as f:
                        youtube_text = f.read().strip()
                    llama_youtube_conversation = generate_llama_conversation(youtube_text, huggingface_token)
                    llama_youtube_filename = os.path.join(folder_name, "llama_youtube_conversation.txt")
                    with open(llama_youtube_filename, "w", encoding="utf-8") as f:
                        f.write(llama_youtube_conversation)
                    st.write(f"Llama YouTube Conversation saved to {llama_youtube_filename}")
                    outfile.write("### Llama YouTube Conversation ###\n")
                    outfile.write(llama_youtube_conversation)
                    outfile.write("\n\n")
                
                if os.path.exists(google_filename):
                    with open(google_filename, "r", encoding="utf-8") as f:
                        google_text = f.read().strip()
                    llama_google_conversation = generate_llama_conversation(google_text, huggingface_token)
                    llama_google_filename = os.path.join(folder_name, "llama_google_conversation.txt")
                    with open(llama_google_filename, "w", encoding="utf-8") as f:
                        f.write(llama_google_conversation)
                    st.write(f"Llama Google Conversation saved to {llama_google_filename}")
                    outfile.write("### Llama Google Conversation ###\n")
                    outfile.write(llama_google_conversation)
                    outfile.write("\n\n")
                
            st.write(f"Combined Llama conversations saved to {combined_llama_filename}")
        else:
            st.write("Please extract YouTube and Google text first to create the necessary files.")

    if st.button("Review Conversations"):
        # 생성된 대화 검토
        folder_name = st.session_state.get('folder_name')
        
        if folder_name:
            gpt_youtube_filename = os.path.join(folder_name, "chatgpt_youtube_conversation.txt")
            gpt_google_filename = os.path.join(folder_name, "chatgpt_google_conversation.txt")
            llama_youtube_filename = os.path.join(folder_name, "llama_youtube_conversation.txt")
            llama_google_filename = os.path.join(folder_name, "llama_google_conversation.txt")
            
            gpt_youtube_review_filename = os.path.join(folder_name, "chatgpt_youtube_conversation_by_llama.txt")
            gpt_google_review_filename = os.path.join(folder_name, "chatgpt_google_conversation_by_llama.txt")
            llama_youtube_review_filename = os.path.join(folder_name, "llama_youtube_conversation_by_chatgpt.txt")
            llama_google_review_filename = os.path.join(folder_name, "llama_google_conversation_by_chatgpt.txt")
            
            combined_gpt_review_filename = os.path.join(folder_name, "combined_chatgpt_conversation_by_llama.txt")
            combined_llama_review_filename = os.path.join(folder_name, "combined_llama_conversation_by_chatgpt.txt")

            with open(combined_gpt_review_filename, "w", encoding="utf-8") as gpt_outfile:
                if os.path.exists(gpt_youtube_filename):
                    process_conversation_with_llama3(gpt_youtube_filename, gpt_youtube_review_filename)
                    st.write(f"GPT YouTube Conversation Review saved to {gpt_youtube_review_filename}")
                    with open(gpt_youtube_review_filename, "r", encoding="utf-8") as f:
                        gpt_outfile.write(f.read())
                        gpt_outfile.write("\n\n")
                if os.path.exists(gpt_google_filename):
                    process_conversation_with_llama3(gpt_google_filename, gpt_google_review_filename)
                    st.write(f"GPT Google Conversation Review saved to {gpt_google_review_filename}")
                    with open(gpt_google_review_filename, "r", encoding="utf-8") as f:
                        gpt_outfile.write(f.read())
                        gpt_outfile.write("\n\n")

            with open(combined_llama_review_filename, "w", encoding="utf-8") as llama_outfile:
                if os.path.exists(llama_youtube_filename):
                    process_conversation_with_chatgpt(llama_youtube_filename, llama_youtube_review_filename)
                    st.write(f"Llama YouTube Conversation Review saved to {llama_youtube_review_filename}")
                    with open(llama_youtube_review_filename, "r", encoding="utf-8") as f:
                        llama_outfile.write(f.read())
                        llama_outfile.write("\n\n")
                if os.path.exists(llama_google_filename):
                    process_conversation_with_chatgpt(llama_google_filename, llama_google_review_filename)
                    st.write(f"Llama Google Conversation Review saved to {llama_google_review_filename}")
                    with open(llama_google_review_filename, "r", encoding="utf-8") as f:
                        llama_outfile.write(f.read())
                        llama_outfile.write("\n\n")

            st.write(f"Combined reviewed GPT conversations saved to {combined_gpt_review_filename}")
            st.write(f"Combined reviewed Llama conversations saved to {combined_llama_review_filename}")

if __name__ == "__main__":
    main()
