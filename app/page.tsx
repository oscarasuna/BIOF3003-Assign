'use client';
import useCamera from './hooks/useCamera';
import SimpleCard from './components/SimpleCard';
import ChartComponent from './components/ChartComponent';
import { useState, useEffect } from 'react';
import usePPGFromSamples from './hooks/usePPGFromSamples';
import { computePPGFromRGB } from './lib/ppg';
import type { SignalCombinationMode } from './components/SignalCombinationSelector';
import SignalCombinationSelector from './components/SignalCombinationSelector';

const SAMPLES_TO_KEEP = 150;

export default function Home() {
  const { videoRef, canvasRef, isRecording, setIsRecording, error } =
    useCamera();
  const [samples, setSamples] = useState<number[]>([]);
  const [apiResponse, setApiResponse] = useState<object | null>(null);
  const { valleys, heartRate, hrv } = usePPGFromSamples(samples);
  const [signalCombination, setSignalCombination] =
    useState<SignalCombinationMode>('default');

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

  return (
    <main className="p-8">
      <h1 className="text-xl font-bold mb-4">Canvas sampling and POST</h1>

      <div className="mb-4">
        <h2 className="text-lg font-semibold mb-2">Camera</h2>
        <div className="w-96 max-w-full border border-gray-400 bg-black min-h-[240px] flex items-center justify-center overflow-hidden rounded">
          <video ref={videoRef} autoPlay muted playsInline className="hidden" />
          {isRecording ? (
            <canvas
              ref={canvasRef}
              className="w-full h-full min-h-[240px] object-contain"
            />
          ) : (
            <span className="text-gray-500 text-sm">
              Start recording to see camera
            </span>
          )}
        </div>
        <div className="mt-2">
          <button
            onClick={() => setIsRecording((r) => !r)}
            className="px-4 py-2 bg-green-500 text-white rounded"
          >
            {isRecording ? 'Stop recording' : 'Start recording'}
          </button>
          {error && <p className="text-red-600 mt-2">{error}</p>}
        </div>
      </div>

      <div className="mt-4">
        <ChartComponent ppgData={samples} valleys={valleys} />
        <SignalCombinationSelector
          value={signalCombination}
          onChange={setSignalCombination}
        />
        <div className="mt-2 flex flex-wrap gap-4">
          <SimpleCard
            title="Heart rate"
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
            title="HRV"
            value={hrv.sdnn > 0 ? `${hrv.sdnn} ms` : '--'}
          />
        </div>
      </div>
      <div className="mt-4 flex flex-wrap gap-4">
        <SimpleCard
          title="Current PPG"
          value={samples[samples.length - 1]?.toFixed(1) ?? '-'}
        />
        <SimpleCard
          title="Last 20"
          value={
            samples
              .slice(-20)
              .map((s) => s.toFixed(0))
              .join(', ') || '-'
          }
        />
      </div>
      <button
        onClick={sendToApi}
        className="mt-4 px-4 py-2 bg-blue-500 text-white rounded"
      >
        Send to API
      </button>
      {apiResponse && (
        <pre className="mt-2 p-2 bg-gray-100 rounded text-sm">
          {JSON.stringify(apiResponse, null, 2)}
        </pre>
      )}
    </main>
  );
}
