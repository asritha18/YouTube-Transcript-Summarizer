from flask import Flask, render_template, request,url_for,flash
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from transformers import BartTokenizer, BartForConditionalGeneration
from urllib.parse import urlparse
from transformers import PegasusTokenizer, PegasusForConditionalGeneration


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.txt']
app.config['UPLOAD_PATH'] = 'uploads'


def validate_youtube_url(url):
    if 'youtube.com' not in url and 'youtu.be' not in url:
        return False
    return True


def validate_url(url):
    try:
        parsed_url = urlparse(url)
        return parsed_url.scheme and parsed_url.netloc
    except TypeError:
        return False


@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    summary = None

    if request.method == 'POST':
        summary_type = request.form.get('type')
        if summary_type == 'youtube':
            url = request.form.get('url')
            if not validate_youtube_url(url):
                error = 'Invalid YouTube URL'
            else:
                try:
                    video_id = url.split("=")[1]
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                    result = ""
                    for i in transcript:
                        result += ' ' + i['text']
                    summarizerfb = pipeline("summarization", model="facebook/bart-large-cnn")
    
                    num_iters = int(len(result)/1000)
                    summarized_text = []
                    summarized_text2 = []
                    for i in range(0, num_iters + 1):
                        start = 0
                        start = i * 1000
                        end = (i + 1) * 1000
                        out = summarizerfb(result[start:end], max_length=130, min_length=30, do_sample=False)
                        out = out[0]
                        out = out['summary_text']
                        summarized_text.append(out)
                        summarized_text2 = ' '.join(summarized_text)
                    summary= (str(summarized_text2))
                except:
                    error = 'Failed to generate summary'
        elif summary_type == 'text':
            text = request.form.get('text')
            if not text:
                error = 'Invalid text input'
            else:
                try:

                    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
                    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

    # perform summarization
                    tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
                    summary = model.generate(**tokens, num_beams=4, length_penalty=2.0, min_length=30, max_length=100)
                    summary = tokenizer.decode(summary[0],skip_special_tokens=True)
                except:
                    error = 'Failed to generate summary'

        elif summary_type == 'url':
            url = request.form.get('url1')
            if not validate_url(url):
                error = 'Invalid URL'
            else:
                try:
                    summarizer = pipeline("summarization")

    # getting our blog post
    
                    r = requests.get(url)
                    soup = BeautifulSoup(r.text, 'html.parser')
                    results = soup.find_all(['h1', 'p'])
                    text = [result.text for result in results]
                    ARTICLE = ' '.join(text)

    # replacing punctuations with end-of-sentence tags
                    ARTICLE = ARTICLE.replace('.', '.')
                    ARTICLE = ARTICLE.replace('?', '?')
                    ARTICLE = ARTICLE.replace('!', '!')
                    sentences = ARTICLE.split(' ')

    # chunking text
                    max_chunk = 500
                    current_chunk = 0 
                    chunks = []
                    for sentence in sentences:
        # checking if we have an empty chunk 
                        if len(chunks) == current_chunk + 1: 
                            if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                                chunks[current_chunk].extend(sentence.split(' '))
                            else:
                                current_chunk += 1
                                chunks.append(sentence.split(' '))
                        else:
                            chunks.append(sentence.split(' '))
                    for chunk_id in range(len(chunks)):
                        chunks[chunk_id] =  ' '.join(chunks[chunk_id])

    # summarizing text
                    res = summarizer(chunks, max_length=70, min_length=30, do_sample=False)
                    summary = ''.join([summ['summary_text'] for summ in res])

                except:
                    error = 'Failed to generate summary'
        elif summary_type == 'file':
            file = request.files['file']
            if not file or file.filename == '':
                error = 'No file selected'
            elif not file.filename.endswith(tuple(app.config['UPLOAD_EXTENSIONS'])):
                error = 'Invalid file type'
            else:
                try:
                    text = file.read().decode('utf-8')
                    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
                    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

    # perform summarization
                    tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
                    summary = model.generate(**tokens, num_beams=4, length_penalty=2.0, min_length=30, max_length=100)
                    summary = tokenizer.decode(summary[0],skip_special_tokens=True)
                    
                except:
                    error = 'Failed to generate summary'


    return render_template('index1.html', error=error, summary=summary)


if __name__ == '__main__':
    app.run(debug=True)