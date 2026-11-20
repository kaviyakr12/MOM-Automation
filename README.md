# MOM Automation

**An LLM-powered application that converts live audio recordings and uploaded audio/video files into structured Minutes of Meeting (MOM). It automates transcription, information extraction, summarization, and MOM generation, reducing manual effort and saving time spent on post-meeting documentation.**


## Prerequisites

- Windows operating system
- Python 3.9 or later
- FFmpeg
- Ollama
- Mistral model
- Microphone (for live audio recording)


## Project Structure

- **Transcription/**
  - `mom.py` – Main Python application and workflow orchestration
  - `mom.html` – Application interface and JavaScript functionality
  - `README.md` – Project documentation

## Technologies Used

- Python                   –        Core application logic and workflow orchestration
- HTML, CSS, JavaScript    –        User interface and interactions
- PyWebView                –        Desktop application interface and Python–JavaScript integration
- FFmpeg                   –        Audio extraction from video files
- Faster-Whisper           –        Speech-to-text transcription
- ThreadPoolExecutor       –        Parallel audio chunk processing
- Regex                    –        Transcript cleaning and text processing
- Ollama (Mistral)         –        Local LLM-based MOM generation


## Features

- **Multiple Input Sources:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Supports live audio recording through microphone and system audio, along with uploaded audio/video files.
- **Audio Preprocessing:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Normalizes audio, removes silence, and divides recordings into manageable chunks with overlapping segments.
- **Parallel Processing:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Processes audio chunks concurrently to improve transcription speed.
- **Speech-to-Text:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Converts audio into text using Faster-Whisper with multilingual support.
- **Transcript Cleaning:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Removes filler words, duplicate content, and unnecessary formatting.
- **LLM-Based Summarization:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Uses Ollama with Mistral to generate structured JSON containing agendas, discussions, decisions, and next steps.
- **Privacy-Focused Processing:**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Uses a local LLM and in-memory transcript processing to help protect confidential meeting information.






