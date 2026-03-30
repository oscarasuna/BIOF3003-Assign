'use client';
import useCamera from './hooks/useCamera';
import SimpleCard from './components/SimpleCard';
import ChartComponent from './components/ChartComponent';
import { useState, useEffect, useRef } from 'react';
import usePPGFromSamples from './hooks/usePPGFromSamples';
import {
  computePPGFromRGB,
  SAMPLES_TO_KEEP,
  MIN_SAMPLES_FOR_DETECTION,
} from './lib/ppg';
import type { SignalCombinationMode } from './components/SignalCombinationSelector';
import SignalCombinationSelector from './components/SignalCombinationSelector';

export default function Home() {
  const { videoRef, canvasRef, isRecording, setIsRecording, error } =
    useCamera();
  const [samples, setSamples] = useState<number[]>([]);
  const [apiResponse, setApiResponse] = useState<object | null>(null);
  const { valleys, heartRate, hrv } = usePPGFromSamples(samples);
  const [signalCombination, setSignalCombination] =
    useState<SignalCombinationMode>('default');

  const [backendStatus, setBackendStatus] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);

  type SegmentLabel = 'good' | 'bad';
  const [segmentLabel, setSegmentLabel] = useState<SegmentLabel>('good');
  const [segmentStatus, setSegmentStatus] = useState<string | null>(null);
  const [labeledSegments, setLabeledSegments] = useState<{ ppgData: number[]; label: string }[]>([]);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  const [inferenceResult, setInferenceResult] = useState<{
    label: string | null;
    confidence: number;
    message?: string;
  } | null>(null);

  const samplesRef = useRef<number[]>([]);
  useEffect(() => {
    samplesRef.current = samples;
  }, [samples]);

  const modelInputRef = useRef<HTMLInputElement>(null);
  const scalerInputRef = useRef<HTMLInputElement>(null);

  const INFERENCE_INTERVAL_MS = 2500;
  useEffect(() => {
    if (!isRecording) return;
    let cancelled = false;
    async function run() {
      const current = samplesRef.current;
      if (current.length < MIN_SAMPLES_FOR_DETECTION) return;
      const segment = current.slice(-SAMPLES_TO_KEEP);
      try {
        const res = await fetch('/api/infer-quality', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ppgData: segment }),
        });
        const data = await res.json();
        if (!cancelled) {
          setInferenceResult({
            label: data.label ?? null,
            confidence: data.confidence ?? 0,
            message: data.message ?? data.error ?? undefined,
          });
        }
      } catch {
        if (!cancelled) {
          setInferenceResult({
            label: null,
            confidence: 0,
            message: 'Request failed',
          });
        }
      }
    }
    run();
    const id = setInterval(run, INFERENCE_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [isRecording]);

  async function checkBackend() {
    try {
      const res = await fetch('/api/health');
      const data = await res.json();
      setBackendStatus(
        data.ok ? 'Backend OK' : 'Backend returned unexpected data',
      );
    } catch (e) {
      setBackendStatus('Backend unreachable');
    }
  }

  async function sendLabeledSegment() {
    if (samples.length < MIN_SAMPLES_FOR_DETECTION) {
      setSegmentStatus('Need more samples (start recording first)');
      return;
    }
    setSegmentStatus(null);
    const ppgSegment = samples.slice(-SAMPLES_TO_KEEP);
    try {
      const res = await fetch('/api/save-labeled-segment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ppgData: ppgSegment, label: segmentLabel }),
      });
      const data = await res.json();
      if (data.success) {
        setLabeledSegments((prev) => [...prev, { ppgData: ppgSegment, label: segmentLabel }]);
        setSegmentStatus(`Saved as ${segmentLabel}`);
      }
      else setSegmentStatus('Error: ' + (data.error || 'Unknown'));
    } catch {
      setSegmentStatus('Error: request failed');
    }
  }

  async function saveRecord() {
    setSaveStatus(null);
    const record = {
      heartRate: { bpm: heartRate.bpm, confidence: heartRate.confidence },
      hrv: {
        sdnn: hrv?.sdnn ?? 0,
        confidence: hrv?.confidence ?? 0,
      },
      ppgData: samples.slice(-SAMPLES_TO_KEEP),
      timestamp: new Date().toISOString(),
    };
    try {
      const res = await fetch('/api/save-record', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(record),
      });
      const data = await res.json();
      if (data.success) setSaveStatus('Saved');
      else setSaveStatus('Error: ' + (data.error || 'Unknown'));
    } catch (e) {
      setSaveStatus('Error: request failed');
    }
  }

  async function sendToApi() {
    const res = await fetch('/api/echo', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        samples: samples.slice(-10),
        timestamp: Date.now(),
      }),
    });
    const data = await res.json();
    setApiResponse(data);
  }

  function downloadLabeledJson() {
    if (labeledSegments.length === 0) return;
    const json = JSON.stringify(labeledSegments, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'labeled_records.json';
    a.click();
    URL.revokeObjectURL(url);
  }

  useEffect(() => {
    const video = videoRef.current;
    const c = canvasRef.current;
    if (!isRecording || !video || !c) return;

    const ctx = c.getContext('2d');
    if (!ctx) return;

    let running = true;
    function tick() {
      if (!running || !ctx) return;
      const v = videoRef.current;
      const c = canvasRef.current;
      if (!v?.srcObject || !v.videoWidth || !c) {
        requestAnimationFrame(tick);
        return;
      }
      c.width = v.videoWidth;
      c.height = v.videoHeight;
      ctx.drawImage(v, 0, 0);
      const w = 10,
        h = 10;
      const x = (c.width - w) / 2;
      const y = (c.height - h) / 2;
      ctx.strokeStyle = 'red';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
      const data = ctx.getImageData(x, y, w, h).data;
      let rSum = 0,
        gSum = 0,
        bSum = 0,
        pixelCount = 0;
      for (let i = 0; i < data.length; i += 4) {
        rSum += data[i];
        gSum += data[i + 1];
        bSum += data[i + 2];
        pixelCount += 1;
      }
      const ppgValue = computePPGFromRGB(
        rSum,
        gSum,
        bSum,
        pixelCount,
        signalCombination,
      );

      setSamples((prev) => [...prev.slice(-(SAMPLES_TO_KEEP - 1)), ppgValue]);

      requestAnimationFrame(tick);
    }
    tick();
    return () => {
      running = false;
    };
  }, [isRecording, signalCombination]);

  async function handleUploadModel(modelFile: File | null, scalerFile: File | null) {
    if (!modelFile || !scalerFile) { setUploadStatus('Select both model and scaler files'); return; }
    setUploadStatus(null);
    try {
      const toBase64 = (f: File) => f.arrayBuffer().then((buf) => btoa(String.fromCharCode(...new Uint8Array(buf))));
      const model = await toBase64(modelFile);
      const scaler = await toBase64(scalerFile);
      const res = await fetch('/api/upload-model', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model, scaler }) });
      const data = await res.json();
      setUploadStatus(res.ok && data.success ? 'Model uploaded' : (data.error || 'Upload failed'));
    } catch {
      setUploadStatus('Upload failed');
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="max-w-7xl mx-auto p-4 md:p-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl md:text-4xl font-bold text-gray-800">
            PPG Signal Quality Analyzer
          </h1>
          <p className="text-gray-600 mt-2">
            Real‑time photoplethysmography (PPG) from webcam with machine learning quality assessment
          </p>
        </div>

        {/* Two‑column grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left column */}
          <div className="space-y-6">
            {/* Camera Card */}
            <div className="bg-white rounded-xl shadow-md overflow-hidden">
              <div className="p-4 border-b border-gray-200">
                <h2 className="text-xl font-semibold text-gray-800">Camera Feed</h2>
              </div>
              <div className="p-4">
                <div className="relative bg-black rounded-lg overflow-hidden aspect-video flex items-center justify-center">
                  <video ref={videoRef} autoPlay muted playsInline className="hidden" />
                  {isRecording ? (
                    <canvas ref={canvasRef} className="w-full h-full object-contain" />
                  ) : (
                    <span className="text-gray-400 text-sm">
                      Start recording to see camera
                    </span>
                  )}
                </div>
                <div className="mt-4">
                  <button
                    onClick={() => setIsRecording((r) => !r)}
                    className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                      isRecording
                        ? 'bg-red-500 hover:bg-red-600 text-white'
                        : 'bg-green-500 hover:bg-green-600 text-white'
                    }`}
                  >
                    {isRecording ? 'Stop Recording' : 'Start Recording'}
                  </button>
                  {error && <p className="text-red-600 mt-2 text-sm">{error}</p>}
                </div>
              </div>
            </div>

            {/* Chart Card */}
            <div className="bg-white rounded-xl shadow-md overflow-hidden">
              <div className="p-4 border-b border-gray-200">
                <h2 className="text-xl font-semibold text-gray-800">PPG Signal</h2>
              </div>
              <div className="p-4">
                <ChartComponent
                  ppgData={samples.slice(-SAMPLES_TO_KEEP)}
                  valleys={valleys}
                />
                <div className="mt-4">
                  <SignalCombinationSelector
                    value={signalCombination}
                    onChange={setSignalCombination}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Right column */}
          <div className="space-y-6">
            {/* Metrics Row */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <SimpleCard
                title="Heart Rate"
                value={heartRate.bpm > 0 ? `${heartRate.bpm} bpm` : '--'}
              />
              <SimpleCard
                title="Confidence"
                value={
                  heartRate.confidence > 0
                    ? `${heartRate.confidence.toFixed(0)}%`
                    : '--'
                }
              />
              <SimpleCard
                title="HRV (SDNN)"
                value={hrv.sdnn > 0 ? `${hrv.sdnn} ms` : '--'}
              />
            </div>

            {/* Current Values Card */}
            <div className="bg-white rounded-xl shadow-md p-4">
              <h3 className="text-lg font-medium text-gray-800 mb-2">Live Values</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-500">Current PPG</p>
                  <p className="text-2xl font-mono font-bold text-gray-800">
                    {samples[samples.length - 1]?.toFixed(1) ?? '-'}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Last 20 samples</p>
                  <p className="text-sm font-mono text-gray-600 truncate">
                    {samples.slice(-20).map((s) => s.toFixed(0)).join(', ') || '-'}
                  </p>
                </div>
              </div>
            </div>

            {/* Actions Card */}
            <div className="bg-white rounded-xl shadow-md p-4">
              <h3 className="text-lg font-medium text-gray-800 mb-3">Actions</h3>
              <div className="flex flex-wrap gap-3">
                <button
                  onClick={checkBackend}
                  className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
                >
                  Check Backend
                </button>
                <button
                  onClick={saveRecord}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                >
                  Save Record
                </button>
              </div>
              {backendStatus && <p className="mt-2 text-sm text-gray-600">{backendStatus}</p>}
              {saveStatus && <p className="mt-2 text-sm text-gray-600">{saveStatus}</p>}
            </div>

            {/* Labeled Data Collection Card */}
            <div className="bg-white rounded-xl shadow-md p-4">
              <h3 className="text-lg font-medium text-gray-800 mb-3">Collect Labeled Data</h3>
              <p className="text-sm text-gray-600 mb-3">
                Label the current signal segment to build a training dataset.
              </p>
              <div className="flex items-center gap-4 mb-3">
                <label className="flex items-center gap-2">
                  <input
                    type="radio"
                    name="segmentLabel"
                    checked={segmentLabel === 'good'}
                    onChange={() => setSegmentLabel('good')}
                    className="text-blue-600"
                  />
                  <span>Good</span>
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="radio"
                    name="segmentLabel"
                    checked={segmentLabel === 'bad'}
                    onChange={() => setSegmentLabel('bad')}
                    className="text-blue-600"
                  />
                  <span>Bad</span>
                </label>
              </div>
              <div className="flex flex-wrap gap-3">
                <button
                  onClick={sendLabeledSegment}
                  className="px-4 py-2 bg-amber-500 hover:bg-amber-600 text-white rounded-lg transition-colors"
                >
                  Send Labeled Segment
                </button>
                <button
                  onClick={downloadLabeledJson}
                  className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg transition-colors disabled:opacity-50"
                  disabled={labeledSegments.length === 0}
                >
                  Download JSON
                </button>
              </div>
              {segmentStatus && <p className="mt-2 text-sm text-gray-600">{segmentStatus}</p>}
            </div>

            {/* Model Upload Card */}
            <div className="bg-white rounded-xl shadow-md p-4">
              <h3 className="text-lg font-medium text-gray-800 mb-3">Upload ML Model</h3>
              <p className="text-sm text-gray-600 mb-3">
                Upload a trained quality model (.joblib) and scaler to use for inference.
              </p>
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Model file</label>
                  <input
                    type="file"
                    ref={modelInputRef}
                    accept=".joblib"
                    className="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Scaler file</label>
                  <input
                    type="file"
                    ref={scalerInputRef}
                    accept=".joblib"
                    className="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                  />
                </div>
                <button
                  onClick={() => handleUploadModel(modelInputRef.current?.files?.[0] ?? null, scalerInputRef.current?.files?.[0] ?? null)}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                >
                  Upload Model and Scaler
                </button>
                {uploadStatus && <p className="text-sm text-gray-600">{uploadStatus}</p>}
              </div>
            </div>

            {/* Inference Result Card */}
            <div className="bg-white rounded-xl shadow-md p-4">
              <h3 className="text-lg font-medium text-gray-800 mb-3">Signal Quality (ML)</h3>
              <p className="text-sm text-gray-600 mb-3">
                Automatic quality assessment using the uploaded model.
              </p>
              <div className="mt-2 text-sm">
                {inferenceResult?.message && (
                  <p className="text-gray-600 bg-yellow-50 p-2 rounded">{inferenceResult.message}</p>
                )}
                {inferenceResult?.label ? (
                  <div className="flex items-center gap-3">
                    <span className="font-medium">Prediction:</span>
                    <span
                      className={`px-3 py-1 rounded-full text-sm font-medium ${
                        inferenceResult.label === 'good'
                          ? 'bg-green-100 text-green-800'
                          : 'bg-red-100 text-red-800'
                      }`}
                    >
                      {inferenceResult.label}
                    </span>
                    {inferenceResult.confidence > 0 && (
                      <span className="text-gray-600">
                        ({(inferenceResult.confidence * 100).toFixed(0)}% confidence)
                      </span>
                    )}
                  </div>
                ) : (
                  <p className="text-gray-500">
                    {isRecording && samples.length < MIN_SAMPLES_FOR_DETECTION
                      ? 'Collecting samples…'
                      : !isRecording
                        ? 'Start recording for quality inference'
                        : '--'}
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}