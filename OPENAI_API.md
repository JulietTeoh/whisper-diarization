Create transcription
post https://api.openai.com/v1/audio/transcriptions

curl https://api.openai.com/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F file="@/path/to/file/audio.mp3" \
  -F model="gpt-4o-transcribe"
{
  "text": "Imagine the wildest idea that you've ever had, and you're curious about how it might scale to something that's a 100, a 1,000 times bigger. This is a place where you can get to do that.",
  "usage": {
    "type": "tokens",
    "input_tokens": 14,
    "input_token_details": {
      "text_tokens": 0,
      "audio_tokens": 14
    },
    "output_tokens": 45,
    "total_tokens": 59
  }
}


Transcribes audio into the input language.

Request body:
file
Required
The audio file object (not file name) to transcribe, in one of these formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.

model
string
Required
ID of the model to use. The options are gpt-4o-transcribe, gpt-4o-mini-transcribe, and whisper-1 (which is powered by our open source Whisper V2 model).

chunking_strategy
"auto" or object
Optional
Controls how the audio is cut into chunks. When set to "auto", the server first normalizes loudness and then uses voice activity detection (VAD) to choose boundaries. server_vad object can be provided to tweak VAD detection parameters manually. If unset, the audio is transcribed as a single block.

include[]
array
Optional
Additional information to include in the transcription response. logprobs will return the log probabilities of the tokens in the response to understand the model's confidence in the transcription. logprobs only works with response_format set to json and only with the models gpt-4o-transcribe and gpt-4o-mini-transcribe.

language
string
Optional
The language of the input audio. Supplying the input language in ISO-639-1 (e.g. en) format will improve accuracy and latency.

prompt
string
Optional
An optional text to guide the model's style or continue a previous audio segment. The prompt should match the audio language.

response_format
string
Optional
Defaults to json
The format of the output, in one of these options: json, text, srt, verbose_json, or vtt. For gpt-4o-transcribe and gpt-4o-mini-transcribe, the only supported format is json.

stream
boolean or null
Optional
Defaults to false
If set to true, the model response data will be streamed to the client as it is generated using server-sent events. See the Streaming section of the Speech-to-Text guide for more information.
Note: Streaming is not supported for the whisper-1 model and will be ignored.

temperature

number
Optional
Defaults to 0
The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. If set to 0, the model will use log probability to automatically increase the temperature until certain thresholds are hit.
timestamp_granularities[]

array
Optional
Defaults to segment
The timestamp granularities to populate for this transcription. response_format must be set verbose_json to use timestamp granularities. Either or both of these options are supported: word, or segment. Note: There is no additional latency for segment timestamps, but generating word timestamps incurs additional latency.

Returns
The transcription object, a verbose transcription object or a stream of transcript events.

The transcription object (JSON)
Represents a transcription response returned by model, based on the provided input.
logprobs
array
The log probabilities of the tokens in the transcription. Only returned with the models gpt-4o-transcribe and gpt-4o-mini-transcribe if logprobs is added to the include array.

    bytes
    array
    The bytes of the token.

    logprob
    number
    The log probability of the token.

    token
    string
    The token in the transcription.

text
string
The transcribed text.

usage
object
Token usage statistics for the request.

{
  "text": "Imagine the wildest idea that you've ever had, and you're curious about how it might scale to something that's a 100, a 1,000 times bigger. This is a place where you can get to do that.",
  "usage": {
    "type": "tokens",
    "input_tokens": 14,
    "input_token_details": {
      "text_tokens": 10,
      "audio_tokens": 4
    },
    "output_tokens": 101,
    "total_tokens": 115
  }
}


The transcription object (Verbose JSON)
Represents a verbose json transcription response returned by model, based on the provided input.

duration
number
The duration of the input audio.

language
string
The language of the input audio.

segments
array
Segments of the transcribed text and their corresponding details.

avg_logprob
number
Average logprob of the segment. If the value is lower than -1, consider the logprobs failed.

compression_ratio
number
Compression ratio of the segment. If the value is greater than 2.4, consider the compression failed.
end

number
End time of the segment in seconds.

id
integer
Unique identifier of the segment.

no_speech_prob
number
Probability of no speech in the segment. If the value is higher than 1.0 and the avg_logprob is below -1, consider this segment silent.

seek
integer
Seek offset of the segment.

start
number
Start time of the segment in seconds.

temperature
number
Temperature parameter used for generating the segment.

text
string
Text content of the segment.

tokens
array
Array of token IDs for the text content.

{
  "task": "transcribe",
  "language": "english",
  "duration": 8.470000267028809,
  "text": "The beach was a popular spot on a hot summer day. People were swimming in the ocean, building sandcastles, and playing beach volleyball.",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 3.319999933242798,
      "text": " The beach was a popular spot on a hot summer day.",
      "tokens": [
        50364, 440, 7534, 390, 257, 3743, 4008, 322, 257, 2368, 4266, 786, 13, 50530
      ],
      "temperature": 0.0,
      "avg_logprob": -0.2860786020755768,
      "compression_ratio": 1.2363636493682861,
      "no_speech_prob": 0.00985979475080967
    },
    ...
  ],
  "usage": {
    "type": "duration",
    "seconds": 9
  }
}
