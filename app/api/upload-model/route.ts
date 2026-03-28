import { NextResponse } from 'next/server';

const flaskUrl = process.env.FLASK_URL || 'http://127.0.0.1:5000';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    if (!body.model || !body.scaler || typeof body.model !== 'string' || typeof body.scaler !== 'string') {
      return NextResponse.json({ success: false, error: 'Missing model or scaler (base64)' }, { status: 400 });
    }
    const res = await fetch(`${flaskUrl}/upload-model`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) return NextResponse.json(data, { status: res.status });
    return NextResponse.json(data);
  } catch {
    return NextResponse.json({ success: false, error: 'Request failed' }, { status: 502 });
  }
}