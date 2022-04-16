from youtube_transcript_api import YouTubeTranscriptApi
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer

def generate_summary(link):    
    def get_transcript(videolink):
        video_id = videolink.split("=")[1]
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        except:
            return "Sorry, failed to summarize the video."
        text = ''
        for value in transcript:
            for key,val in value.items():
                if key == 'text':
                    text += val + ' '

        split_text = text.splitlines()
        transcript = " ".join(split_text)
        return transcript
    
    def get_summary(transcript):
        summary = ''
        parser = PlaintextParser.from_string(transcript,Tokenizer("english"))
        summarizer_lsa = LsaSummarizer()
        summary_list = summarizer_lsa(parser.document, 1)
        for sentence in summary_list:
            summary += str(sentence)
        return summary
    

    transcript = get_transcript(link)
    summary = get_summary(transcript)
    
    return summary
