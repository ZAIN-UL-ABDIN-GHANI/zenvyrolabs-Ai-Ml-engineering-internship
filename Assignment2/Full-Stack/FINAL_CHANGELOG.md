# Final Changelog — Zenvyrolabs Advanced Voice Studio

Production-readiness pass across a 6-layer architecture (`voice_studio/`) built incrementally on top of the original delivered application. Original files were extended additively; no existing tab, function signature, or return contract was removed.

## Architecture Added
- **Domain**: VoiceProfile, DialogueLine/PodcastScript, Language/LanguageDetector, PronunciationPolicy, TrainingSession, ModelArtifact
- **Application**: PodcastScriptParser, PronunciationRoutingService, PodcastEngine
- **Storage**: VoiceStoreRepository, ModelStoreRepository (path-traversal-safe, backward-compatible with existing `saved_voices/`)
- **Config**: centralized `VoiceStudioSettings` (paths/engine/feature flags, env-overridable)
- **Infrastructure**: ProcessRunner, structured JSON logging + correlation IDs, DependencyHealthChecker
- **AI Processing**: VoiceGenerationEngine (F5-TTS), NeuralNarrationEngine (Edge-TTS), VoiceConversionEngine (RVC), TranscriptionEngine (Whisper), SpeakerSimilarityAnalyzer (Resemblyzer)
- **Audio Processing**: AudioStitcher (crossfade), TrainingAudioPreprocessor (normalize → denoise → trim silence → chunk)

## Original Findings Closed
F-01 (detected via integrity check), F-04, F-06 (path traversal), F-07, F-09, F-10, F-11, F-12, F-13, F-14, F-15, F-16, F-18, F-19 (Resemblyzer), F-20 (resolved as a side effect), F-21, F-22, F-23, F-26.

## Task Completion
- **Task 1 (Docker)**: Dockerfile + docker-compose.yml, dual-venv (matches existing subprocess architecture), FFmpeg, persistent named volumes for all 5 stores, one-command `docker compose up -d`, no `setup.bat` dependency.
- **Task 2 (Podcast Engine)**: parser handles all specified spacing variants without crashing; unknown speakers produce friendly warnings; pronunciation routing verified for pure/mixed Hindi, Urdu, and Roman variants with zero false positives on English (including YouTube-style/technical/pangram text); crossfaded transitions confirmed clip-free even at near-maximum amplitude.
- **Task 3 (Training Studio)**: noise filtering, silence removal, normalization, and chunking verified on clean/noisy/silent/invalid/12-minute audio, all with graceful, readable error handling.

## This Session's Production Review
- Found and fixed a real language-detection gap: mixed Hindi-English/Urdu-English sentences with a low proportion of Hindi words fell under the match-ratio threshold. Root cause: `COMMON_WORDS` (correct for transliteration) contains English loanwords (`subscribe`, `the`, `do`, `me`, `main`, `log`, etc.) that are poor detection signals. Fixed by separating a detection-specific vocabulary (exclusion list) from the transliteration vocabulary (untouched), and re-tuned thresholds — verified against 19 combined must-route/must-not-route cases plus 48 new regression tests.
- Found and fixed `AudioStitcher.stitch()` not coercing string output paths (a `str`-typed caller would crash) — now defensively coerces, matching the pattern already used elsewhere.
- Verified: no clipping in crossfaded output, no orphaned modules, no TODO/FIXME/debug code, no stray print statements.
- Test suite grew from 134 → 176 tests across this review.

## Known, Documented Non-Blockers
- F-02 (rvc_infer.py API shape) and F-03 (missing `download_gojo.py`) require external package verification / content this review cannot fabricate — flagged in `requirements-rvc.txt`, not silently assumed fixed.
- F-05 (MP3-mislabeled-`.wav`) is handled gracefully by every code path (pydub content-sniffs) and was not altered.
- Actual model *training* (fitting new RVC/F5-TTS weights) was never in the original scope of any of the 4 tasks — this pipeline produces training-ready datasets, matching the original Task 3 wording exactly.

## Dependencies
`requirements.txt`: added `tomli_w`, `pydub`, `noisereduce`, `f5-tts`, `edge-tts`, `transformers`, `soundfile`, `resemblyzer`, `setuptools<81`; removed `TTS` (dead, abandoned upstream) and `yt-dlp` (unused).
`requirements-rvc.txt`: new — the previously entirely undocumented isolated-runtime dependency manifest.
