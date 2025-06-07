# evals.py
import json
import os
from typing import Dict, List, Any
from datetime import datetime
import webbrowser
import threading
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from i18n import translation_prompt, Language
from server import translate_text, text_to_speech, client, STT_MODEL


# Create eval data directory
EVAL_DATA_DIR = ".vscode/eval-data"
os.makedirs(EVAL_DATA_DIR, exist_ok=True)


def transcribe_audio(audio_file_path: str) -> Dict[str, Any]:
    """Transcribe audio file using ElevenLabs Speech-to-Text API"""
    try:
        timestamp = int(time.time())

        # Read audio file
        with open(audio_file_path, "rb") as audio_file:
            audio_buffer = audio_file.read()

        # Use ElevenLabs STT
        result = client.speech_to_text.convert(
            model_id=STT_MODEL,
            file=(f"input-file-{timestamp}.wav", audio_buffer, "audio/wav"),
            timestamps_granularity="word",
        )

        # Extract text from result
        # Note: You may need to adjust this based on ElevenLabs response format
        transcribed_text = result.text if hasattr(result, "text") else str(result)

        return {
            "text": transcribed_text,
            "language": "unknown",  # ElevenLabs may not provide language detection
            "success": True,
            "error": None,
            "raw_result": result.model_dump(),
        }
    except Exception as e:
        return {"text": None, "language": None, "success": False, "error": str(e), "raw_result": None}


def prepare_test_data_from_audio() -> List[Dict[str, Any]]:
    """Prepare test data by transcribing test_audio.wav"""
    audio_file_path = "test_audio.wav"

    if not os.path.exists(audio_file_path):
        print(f"Error: {audio_file_path} not found!")
        return []

    print(f"Transcribing {audio_file_path} using ElevenLabs...")
    transcription_result = transcribe_audio(audio_file_path)

    if not transcription_result["success"]:
        print(f"Fatal error: Transcription failed: {transcription_result['error']}")
        exit()
        # return[]

    print("Transcription successful!")
    print(f"Transcribed text: {transcription_result['text']}")
    # Since ElevenLabs may not provide language detection,
    # you'll need to specify or detect the language manually
    detected_lang = "ENGLISH"  # Default assumption, or add language detection logic

    # Create test data from transcription
    test_data = [
        {
            "source_text": transcription_result["text"],
            "source_lang": detected_lang,
            "context": "transcribed_audio_elevenlabs",
            "original_audio_file": audio_file_path,
            "stt_provider": "elevenlabs",
            "raw_transcription_result": transcription_result.get("raw_result"),
        }
    ]

    return test_data


TARGET_LANGUAGES = ["CHINESE", "VIETNAMESE", "THAI", "ENGLISH"]


class TranslationEvaluator:
    def __init__(self):
        self.results = []
        self.evaluation_data = []
        self.test_data = []
        self.tts_data = []

    def prepare_test_data(self):
        """Prepare test data from audio transcription"""
        self.test_data = prepare_test_data_from_audio()
        if not self.test_data:
            print("No test data available. Using fallback data.")
            # Fallback test data if audio transcription fails
            self.test_data = [{"source_text": "Hello, how are you today?", "source_lang": "ENGLISH", "context": "fallback_greeting"}]

    def evaluate_prompt_format(self) -> Dict[str, Any]:
        """Evaluate the format and structure of translation prompts"""
        format_results = {"total_prompts": len(translation_prompt), "format_issues": [], "missing_pairs": [], "valid_prompts": 0}

        # Check all possible language pairs
        languages = [Language.ENGLISH, Language.CHINESE, Language.VIETNAMESE, Language.THAI]
        expected_pairs = []

        for source in languages:
            for target in languages:
                if source != target:
                    expected_pairs.append((source, target))

        # Check for missing language pairs
        for pair in expected_pairs:
            if pair not in translation_prompt:
                format_results["missing_pairs"].append(f"{pair[0].value} -> {pair[1].value}")

        # Validate existing prompts
        for key, prompt in translation_prompt.items():
            if not isinstance(prompt, str):
                format_results["format_issues"].append(f"Prompt for {key} is not a string")
            elif "${text}" not in prompt:
                format_results["format_issues"].append(f"Prompt for {key} missing " + "${text} placeholder")
            elif len(prompt.strip()) == 0:
                format_results["format_issues"].append(f"Prompt for {key} is empty")
            else:
                format_results["valid_prompts"] += 1

        return format_results

    def generate_tts_audio(self, text: str, target_lang: str, eval_id: str) -> str:
        """Generate TTS audio and save to eval data directory"""
        try:
            print(f"    Generating TTS audio for {target_lang}...")
            audio_bytes = text_to_speech(text, target_lang)

            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tts_{eval_id}_{target_lang.lower()}_{timestamp}.mp3"
            filepath = os.path.join(EVAL_DATA_DIR, filename)

            # Save audio file
            with open(filepath, "wb") as f:
                f.write(audio_bytes)

            print(f"      ✓ TTS audio saved: {filename}")
            return filename

        except Exception as e:
            print(f"      ✗ TTS failed: {str(e)}")
            return None

    def evaluate_translation_recall(self) -> List[Dict[str, Any]]:
        """Test translation functionality with transcribed test data and generate TTS"""
        recall_results = []

        for test_idx, test_case in enumerate(self.test_data):
            source_text = test_case["source_text"]
            source_lang = test_case["source_lang"]
            context = test_case["context"]

            print(f"\nTesting translations for: '{source_text[:50]}...'")
            print(f"Source language: {source_lang}")

            for target_lang in TARGET_LANGUAGES:
                if source_lang == target_lang:
                    continue

                eval_id = f"test{test_idx}_{source_lang.lower()}_{target_lang.lower()}"

                try:
                    print(f"  Translating to {target_lang}...")
                    # Test translation
                    start_time = time.time()
                    translated_text = translate_text(source_text, source_lang, target_lang)
                    translation_time = time.time() - start_time

                    # Generate TTS audio for the translated text
                    tts_filename = self.generate_tts_audio(translated_text, target_lang, eval_id)

                    result = {
                        "eval_id": eval_id,
                        "source_text": source_text,
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "translated_text": translated_text,
                        "context": context,
                        "translation_time": translation_time,
                        "tts_filename": tts_filename,
                        "tts_filepath": os.path.join(EVAL_DATA_DIR, tts_filename) if tts_filename else None,
                        "success": True,
                        "error": None,
                        "timestamp": datetime.now().isoformat(),
                    }

                    # Add original audio info if available
                    if "original_audio_file" in test_case:
                        result["original_audio_file"] = test_case["original_audio_file"]

                    print(f"    ✓ Success: '{translated_text[:50]}...' ({translation_time:.2f}s)")

                    # Add to evaluation data for potential future human review
                    self.evaluation_data.append(result)

                    # Store TTS data for visualization
                    if tts_filename:
                        self.tts_data.append(
                            {
                                "eval_id": eval_id,
                                "text": translated_text,
                                "language": target_lang,
                                "filename": tts_filename,
                                "source_text": source_text,
                                "source_lang": source_lang,
                            }
                        )

                except Exception as e:
                    result = {
                        "eval_id": eval_id,
                        "source_text": source_text,
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "translated_text": None,
                        "context": context,
                        "translation_time": None,
                        "tts_filename": None,
                        "tts_filepath": None,
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                    print(f"    ✗ Failed: {str(e)}")

                recall_results.append(result)

        return recall_results

    def evaluate_transcription_quality(self) -> Dict[str, Any]:
        """Evaluate the quality of the transcription"""
        if not self.test_data:
            return {"error": "No transcription data available"}

        transcription_eval = {
            "transcribed_text": self.test_data[0]["source_text"],
            "stt_provider": self.test_data[0].get("stt_provider", "elevenlabs"),
            "mapped_language": self.test_data[0]["source_lang"],
            "text_length": len(self.test_data[0]["source_text"]),
            "word_count": len(self.test_data[0]["source_text"].split()),
            "has_punctuation": any(char in self.test_data[0]["source_text"] for char in ".,!?;:"),
            "audio_file": self.test_data[0].get("original_audio_file", "unknown"),
            "raw_result_available": self.test_data[0].get("raw_transcription_result") is not None,
        }

        return transcription_eval


def create_visualization_server(evaluation_results: Dict[str, Any]):
    """Create FastAPI server for visualizing evaluation results"""
    app = FastAPI(title="Translation Evaluation Results")

    # Check if templates directory exists
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    if not os.path.exists(templates_dir):
        raise FileNotFoundError(f"Templates directory not found: {templates_dir}")

    # Check if required template exists
    template_file = os.path.join(templates_dir, "eval.html")
    if not os.path.exists(template_file):
        raise FileNotFoundError(f"Required template not found: {template_file}")

    templates = Jinja2Templates(directory=templates_dir)

    # Mount static files for audio serving
    app.mount("/audio", StaticFiles(directory=EVAL_DATA_DIR), name="audio")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse("eval.html", {"request": request, "results": evaluation_results, "eval_data_dir": EVAL_DATA_DIR})

    @app.get("/api/results")
    async def api_results():
        return evaluation_results

    @app.get("/audio/{filename}")
    async def serve_audio(filename: str):
        file_path = os.path.join(EVAL_DATA_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Audio file not found")
        return FileResponse(file_path)

    @app.post("/api/rate")
    async def rate_audio(request: Request):
        # This endpoint would handle rating submissions
        # Implementation would save ratings to a file or database
        data = await request.json()
        return {"status": "success", "message": "Rating saved", "data": data}

    return app


def start_visualization_server(evaluation_results: Dict[str, Any], port: int = 5000):
    """Start the visualization server in a separate thread"""
    app = create_visualization_server(evaluation_results)

    def run_server():
        print(f"\n🌐 Starting evaluation visualization server on http://localhost:{port}")
        print("   Use this interface to listen to generated TTS audio and rate translations")
        uvicorn.run(app, host="localhost", port=port, log_level="warning")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Give server time to start
    time.sleep(3)  # Increased time for FastAPI startup

    # Open browser
    try:
        webbrowser.open(f"http://localhost:{port}")
        print("   Browser opened automatically to view results")
    except Exception:
        print(f"   Please open http://localhost:{port} in your browser to view results")

    return server_thread


def run_evaluation(with_server: bool = True):
    """Main evaluation function"""
    print("Starting Translation Evaluation with Audio Transcription and TTS Generation...")
    print("=" * 80)

    # Check if templates exist if server is requested
    if with_server:
        templates_dir = os.path.join(os.path.dirname(__file__), "templates")
        template_file = os.path.join(templates_dir, "eval.html")

        if not os.path.exists(templates_dir) or not os.path.exists(template_file):
            print("❌ Template Error: Required templates not found")
            print(f"   Templates directory: {templates_dir}")
            print(f"   Required template: {template_file}")
            print("   Please create the templates directory and add eval.html")
            print("   Running evaluation without visualization server...")
            with_server = False

    evaluator = TranslationEvaluator()

    # 0. Prepare test data from audio transcription
    print("0. Preparing test data from audio transcription...")
    evaluator.prepare_test_data()

    if not evaluator.test_data:
        print("Failed to prepare test data. Exiting.")
        return

    # 0.1. Evaluate transcription quality
    print("\n0.1. Evaluating transcription quality...")
    transcription_eval = evaluator.evaluate_transcription_quality()

    print(f"   Audio file: {transcription_eval.get('audio_file', 'N/A')}")
    print(f"   STT Provider: {transcription_eval.get('stt_provider', 'N/A')}")
    print(f"   Mapped to: {transcription_eval.get('mapped_language', 'N/A')}")
    print(f"   Text length: {transcription_eval.get('text_length', 0)} characters")
    print(f"   Word count: {transcription_eval.get('word_count', 0)} words")
    print(f"   Has punctuation: {transcription_eval.get('has_punctuation', False)}")
    print(f"   Transcribed text: '{transcription_eval.get('transcribed_text', '')}'")

    # 1. Evaluate prompt format
    print("\n1. Evaluating prompt format...")
    format_results = evaluator.evaluate_prompt_format()

    print(f"   Total prompts: {format_results['total_prompts']}")
    print(f"   Valid prompts: {format_results['valid_prompts']}")

    if format_results["format_issues"]:
        print("   Format issues found:")
        for issue in format_results["format_issues"]:
            print(f"     - {issue}")

    if format_results["missing_pairs"]:
        print("   Missing language pairs:")
        for pair in format_results["missing_pairs"]:
            print(f"     - {pair}")

    # 2. Evaluate translation recall and generate TTS
    print("\n2. Evaluating translation recall and generating TTS audio...")
    recall_results = evaluator.evaluate_translation_recall()

    successful_translations = sum(1 for r in recall_results if r["success"])
    total_translations = len(recall_results)
    successful_tts = sum(1 for r in recall_results if r.get("tts_filename"))

    print("\n   Translation Summary:")
    print(f"   Successful translations: {successful_translations}/{total_translations}")
    print(f"   Success rate: {(successful_translations/total_translations)*100:.1f}%")
    print(f"   TTS audio files generated: {successful_tts}")

    # Show successful translations with TTS
    successful_results = [r for r in recall_results if r["success"]]
    if successful_results:
        print("\n   Successful Translations with TTS:")
        for result in successful_results:
            tts_status = "✓ TTS" if result.get("tts_filename") else "✗ No TTS"
            print(f"     {result['source_lang']} → {result['target_lang']}: '{result['translated_text'][:60]}...' [{tts_status}]")

    # Show failed translations
    failed_translations = [r for r in recall_results if not r["success"]]
    if failed_translations:
        print("\n   Failed translations:")
        for failure in failed_translations:
            print(f"     - {failure['source_lang']} → {failure['target_lang']}: {failure['error']}")

    # 3. Save detailed results
    print("\n3. Saving evaluation results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(EVAL_DATA_DIR, f"evaluation_results_{timestamp}.json")

    complete_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "eval_data_dir": EVAL_DATA_DIR,
            "total_translations": total_translations,
            "successful_translations": successful_translations,
            "success_rate": (successful_translations / total_translations) * 100 if total_translations > 0 else 0,
            "tts_files_generated": successful_tts,
        },
        "transcription_evaluation": transcription_eval,
        "format_evaluation": format_results,
        "recall_evaluation": recall_results,
        "evaluation_data": evaluator.evaluation_data,
        "tts_data": evaluator.tts_data,
        "test_data_used": evaluator.test_data,
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(complete_results, f, indent=2, ensure_ascii=False)

    print(f"   Detailed results saved to: {results_file}")
    print(f"   TTS audio files saved in: {EVAL_DATA_DIR}")

    # 4. Create evaluation summary
    print("\n4. Creating evaluation summary...")
    summary_file = os.path.join(EVAL_DATA_DIR, f"evaluation_summary_{timestamp}.txt")

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("TRANSLATION EVALUATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Audio Source: {transcription_eval.get('audio_file', 'N/A')}\n")
        f.write(f"Source Text: {transcription_eval.get('transcribed_text', 'N/A')}\n")
        f.write(f"Source Language: {transcription_eval.get('mapped_language', 'N/A')}\n\n")

        f.write("TRANSLATION RESULTS:\n")
        f.write("-" * 20 + "\n")
        for result in successful_results:
            f.write(f"{result['source_lang']} → {result['target_lang']}:\n")
            f.write(f"  Text: {result['translated_text']}\n")
            f.write(f"  Time: {result['translation_time']:.2f}s\n")
            f.write(f"  TTS: {result.get('tts_filename', 'Not generated')}\n\n")

        f.write("\nSUMMARY STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total translations attempted: {total_translations}\n")
        f.write(f"Successful translations: {successful_translations}\n")
        f.write(f"Success rate: {(successful_translations/total_translations)*100:.1f}%\n")
        f.write(f"TTS files generated: {successful_tts}\n")

    print(f"   Summary saved to: {summary_file}")

    print("\nEvaluation complete!")

    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Audio transcription: {'✓ Success' if transcription_eval.get('transcribed_text') else '✗ Failed'}")
    print(f"Prompt format: {format_results['valid_prompts']}/{format_results['total_prompts']} valid")
    print(f"Translation success rate: {(successful_translations/total_translations)*100:.1f}%")
    print(f"TTS audio files generated: {successful_tts}")
    print(f"Results saved to: {results_file}")
    print(f"Audio files in: {EVAL_DATA_DIR}")

    # 5. Start visualization server if requested
    if with_server:
        print("\n" + "=" * 80)
        print("STARTING EVALUATION VISUALIZATION SERVER")
        print("=" * 80)

        try:
            # Start the visualization server
            start_visualization_server(complete_results)

            print("\n📊 Evaluation visualization server is running!")
            print("   - View results and listen to TTS audio")
            print("   - Rate translation quality")
            print("   - Compare original vs translated text")
            print("\n⏹️  Press Ctrl+C to stop the server and exit")

            # Keep the main thread alive to maintain the server
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n\n🛑 Shutting down evaluation server...")
                print("Evaluation complete. Results saved in:", EVAL_DATA_DIR)

        except FileNotFoundError as e:
            print(f"❌ Template Error: {e}")
            print("   Evaluation completed but visualization server could not start")
            print(f"   Results still saved to: {results_file}")
    else:
        print("\n✅ Evaluation completed without visualization server")
        print(f"   Results saved to: {results_file}")
        print(f"   Audio files saved in: {EVAL_DATA_DIR}")

    return complete_results


if __name__ == "__main__":
    # Check if templates exist to determine if we can run with server
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    template_file = os.path.join(templates_dir, "eval.html")

    templates_exist = os.path.exists(templates_dir) and os.path.exists(template_file)

    if not templates_exist:
        print("⚠️  Warning: Templates not found - running evaluation without visualization server")
        print(f"   Templates directory: {templates_dir}")
        print(f"   Required template: {template_file}")
        print("   To enable visualization server, create the templates directory and add eval.html\n")

    # Run evaluation with or without server based on template availability
    run_evaluation(with_server=templates_exist)
