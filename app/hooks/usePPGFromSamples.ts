'use client';

import { useState, useEffect } from 'react';
import {
  detectValleys,
  heartRateFromValleys,
  hrvFromValleys,
  FPS,
  SAMPLES_TO_KEEP,
  MIN_SAMPLES_FOR_DETECTION,
} from '../lib/ppg';
import type { Valley, HeartRateResult, HRVResult } from '../types';

export default function usePPGFromSamples(samples: number[]) {
  const [valleys, setValleys] = useState<Valley[]>([]);
  const [heartRate, setHeartRate] = useState<HeartRateResult>({
    bpm: 0,
    confidence: 0,
  });
  const [hrv, setHrv] = useState<HRVResult>({ sdnn: 0, confidence: 0 });

  useEffect(() => {
    if (samples.length < MIN_SAMPLES_FOR_DETECTION) return;
    const toUse = samples.slice(-SAMPLES_TO_KEEP);
    const v = detectValleys(toUse, FPS);
    setValleys(v);
    setHeartRate(heartRateFromValleys(v, FPS));
    setHrv(hrvFromValleys(v, FPS));
  }, [samples]);

  return { valleys, heartRate, hrv };
}
