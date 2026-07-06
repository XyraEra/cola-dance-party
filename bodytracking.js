// =============================================================================
// bodytracking.js
// -----------------------------------------------------------------------------
// Wraps MediaPipe Tasks-Vision (PoseLandmarker + HandLandmarker) into a single
// BodyTracker class that turns raw webcam landmarks into discrete, game-ready
// gesture EVENTS. This file knows nothing about soda, customers, or scoring —
// game.js consumes it purely through events. That separation is deliberate:
// you can retune/rewrite gesture detection here without touching game logic.
//
// PUBLIC API
// -----------------------------------------------------------------------------
//   const tracker = new BodyTracker(videoEl, { ...optional overrides of CONFIG });
//   await tracker.init();           // loads models + starts webcam
//   tracker.start();                // begins the detection loop
//   await tracker.calibrate();      // ~1.2s "stand still" baseline capture
//   tracker.stop();                 // pauses loop + stops webcam tracks
//   tracker.dispose();              // stop() + frees MediaPipe/WASM memory
//
// EVENTS (tracker.addEventListener(name, e => ...), payload in e.detail)
// -----------------------------------------------------------------------------
//   'ready'       -> models loaded, webcam streaming. No payload.
//   'calibrated'  -> baseline captured. detail = { shoulderMidX, shoulderWidth, handSize }
//   'found'       -> a body re-entered frame after being lost. No payload.
//   'lost'        -> no body detected for CONFIG.trackingLostMs. No payload.
//   'pose'        -> fires every processed frame (post-smoothing), for drawing
//                    a debug skeleton or a "lean meter" UI.
//                    detail = { pose: Landmark[33], hands: Landmark[21][], handedness }
//   'gesture'     -> a discrete action was detected. detail.type is one of:
//                      'lean'          detail = { direction: 'left'|'right'|'center' }
//                      'raiseHand'     detail = { side: 'left'|'right' }
//                      'clap'          detail = {}
//                      'shake'         detail = {}
//                      'handsForward'  detail = {}
//
// A NOTE ON LEFT/RIGHT
// -----------------------------------------------------------------------------
// MediaPipe labels pose/hand landmarks as if describing someone else's body
// from the camera's viewpoint. Since your webcam faces you like a mirror,
// what MediaPipe calls the subject's "right side" is actually YOUR left side.
// CONFIG.invertHandedness (default true) flips this so 'raiseHand: right'
// really means the player's own right hand. If left/right ever feel swapped
// during playtesting, that's the flag to toggle first.
// =============================================================================

import {
  FilesetResolver,
  PoseLandmarker,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const POSE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// ---- Tunable thresholds ----------------------------------------------------
// Everything gameplay-feel-related lives here so you can retune without
// hunting through detector logic. Values are fractions of body-relative
// measurements (shoulder width, torso length) so they scale with distance
// from the camera and different body sizes.
export const CONFIG = {
  targetFps: 30, // throttle inference; webcams/rAF often run faster than needed
  smoothingAlpha: 0.5, // 0..1, higher = snappier but jitterier

  leanEnterThreshold: 0.35, // shoulder-mid offset (x shoulder widths) to trigger
  leanExitThreshold: 0.18, // must return inside this to count as "center" again
  leanHoldMs: 150, // must be sustained this long before it fires

  raiseThreshold: 0.35, // wrist above shoulder, x torso lengths
  raiseHoldMs: 100,
  raiseCooldownMs: 500,

  clapApartThreshold: 0.9, // wrists must have been at least this far apart (x shoulder widths)...
  clapDistanceThreshold: 0.35, // ...then come together closer than this to count as a clap
  clapCooldownMs: 400,

  shakeWindowMs: 700, // rolling window used to detect oscillation
  shakeMinReversals: 3, // direction changes required inside the window
  shakeAmplitude: 0.12, // min side-to-side range, x shoulder widths
  shakeCooldownMs: 600,

  forwardScaleFactor: 1.5, // hands must appear this much bigger than baseline
  forwardCooldownMs: 500,

  trackingLostMs: 800, // no pose for this long -> emit 'lost'

  mirrored: true, // is the <video>/<canvas> shown flipped via CSS (recommended)?
  invertHandedness: true, // see "A NOTE ON LEFT/RIGHT" above
};

function lerp(a, b, t) {
  return a + (b - a) * t;
}
function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}
function bboxDiagonal(points) {
  let minX = 1, maxX = 0, minY = 1, maxY = 0;
  for (const p of points) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  return Math.hypot(maxX - minX, maxY - minY);
}

export class BodyTracker extends EventTarget {
  constructor(videoEl, userConfig = {}) {
    super();
    this.video = videoEl;
    this.config = { ...CONFIG, ...userConfig };

    // See "A NOTE ON LEFT/RIGHT" — these indices already point at the
    // PLAYER's true anatomical side, not MediaPipe's raw camera-facing labels.
    this.SIDE = this.config.invertHandedness
      ? { L_SHOULDER: 12, R_SHOULDER: 11, L_WRIST: 16, R_WRIST: 15, L_HIP: 24, R_HIP: 23 }
      : { L_SHOULDER: 11, R_SHOULDER: 12, L_WRIST: 15, R_WRIST: 16, L_HIP: 23, R_HIP: 24 };

    this.pose = null;
    this.hands = null;
    this.running = false;
    this.baseline = null;

    this._smoothed = {};
    this._lastInferenceTime = 0;
    this._lastSeenTime = 0;
    this._wasTracking = false;
    this._lastFrame = null;

    // Per-gesture detector state
    this._lean = { state: "center", pending: null, pendingSince: 0 };
    this._raise = {
      left: { raised: false, since: 0, last: 0 },
      right: { raised: false, since: 0, last: 0 },
    };
    this._clap = { wasApart: true, last: 0 };
    this._shake = { buffer: [], last: 0 };
    this._forward = { last: 0 };

    this._loop = this._loop.bind(this);
  }

  async init() {
    try {
      const vision = await FilesetResolver.forVisionTasks(WASM_URL);
      const built = await this._createLandmarkers(vision);
      this.pose = built.pose;
      this.hands = built.hands;
    } catch (err) {
      throw new Error(`BodyTracker: failed to load MediaPipe models — ${err.message}`);
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
      });
      this.video.srcObject = stream;
      await new Promise((resolve) => (this.video.onloadedmetadata = resolve));
      await this.video.play();
    } catch (err) {
      throw new Error(`BodyTracker: camera access failed — ${err.message}`);
    }

    this.dispatchEvent(new CustomEvent("ready"));
  }

  // Tries GPU first (fast, but unsupported on some browsers/OSes), falls
  // back to CPU/WASM so the game still runs everywhere, just a bit slower.
  async _createLandmarkers(vision) {
    const delegates = ["GPU", "CPU"];
    let lastErr;
    for (const delegate of delegates) {
      try {
        const pose = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate },
          runningMode: "VIDEO",
          numPoses: 1,
        });
        const hands = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate },
          runningMode: "VIDEO",
          numHands: 2,
        });
        return { pose, hands, delegate };
      } catch (err) {
        lastErr = err;
        console.warn(`BodyTracker: "${delegate}" delegate failed, trying next...`, err);
      }
    }
    throw lastErr;
  }

  start() {
    if (this.running) return;
    this.running = true;
    requestAnimationFrame(this._loop);
  }

  stop() {
    this.running = false;
    const stream = this.video.srcObject;
    if (stream) stream.getTracks().forEach((t) => t.stop());
  }

  dispose() {
    this.stop();
    this.pose?.close();
    this.hands?.close();
  }

  // Captures a short "stand still / hands at chest height" baseline so
  // lean/raise/clap/forward thresholds adapt to this player's size and
  // distance from the camera. Call at the start of each play session
  // (re-calibrating between rounds isn't necessary unless the player moves).
  calibrate(durationMs = 1200) {
    return new Promise((resolve) => {
      const samples = [];
      const startedAt = performance.now();
      const collect = () => {
        if (this._lastFrame) samples.push(this._lastFrame);
        if (performance.now() - startedAt < durationMs) {
          requestAnimationFrame(collect);
        } else {
          this.baseline = this._computeBaseline(samples);
          this.dispatchEvent(new CustomEvent("calibrated", { detail: this.baseline }));
          resolve(this.baseline);
        }
      };
      collect();
    });
  }

  _computeBaseline(samples) {
    const valid = samples.filter((s) => s.pose);
    const n = Math.max(1, valid.length);
    let shoulderMidX = 0, shoulderWidth = 0, handSize = 0, handCount = 0;

    for (const s of valid) {
      const ls = s.pose[this.SIDE.L_SHOULDER];
      const rs = s.pose[this.SIDE.R_SHOULDER];
      shoulderMidX += (ls.x + rs.x) / 2;
      shoulderWidth += Math.abs(ls.x - rs.x);
      for (const h of s.hands) {
        handSize += bboxDiagonal(h);
        handCount++;
      }
    }

    return {
      shoulderMidX: shoulderMidX / n,
      shoulderWidth: Math.max(0.05, shoulderWidth / n),
      handSize: handCount ? handSize / handCount : null,
    };
  }

  _loop(now) {
    if (!this.running) return;
    const minInterval = 1000 / this.config.targetFps;
    if (now - this._lastInferenceTime >= minInterval) {
      this._lastInferenceTime = now;
      this._processFrame(now);
    }
    requestAnimationFrame(this._loop);
  }

  _processFrame(now) {
    const poseResult = this.pose.detectForVideo(this.video, now);
    const handResult = this.hands.detectForVideo(this.video, now);
    const landmarks = poseResult.landmarks?.[0];

    if (!landmarks) {
      this._handleTrackingLost(now);
      return;
    }

    this._lastSeenTime = now;
    if (!this._wasTracking) {
      this._wasTracking = true;
      this.dispatchEvent(new CustomEvent("found"));
    }

    const pose = this._smooth("pose", landmarks);
    const hands = (handResult.landmarks || []).map((lm, i) => this._smooth(`hand${i}`, lm));
    const handedness = (handResult.handedness || []).map((cats) =>
      cats.map((c) => ({
        ...c,
        categoryName:
          this.config.invertHandedness
            ? c.categoryName === "Left" ? "Right" : "Left"
            : c.categoryName,
      }))
    );

    this._lastFrame = { pose, hands, handedness };
    this.dispatchEvent(new CustomEvent("pose", { detail: this._lastFrame }));

    if (!this.baseline) return; // gestures need a baseline first

    this._detectLean(pose, now);
    this._detectRaise(pose, now);
    this._detectClap(pose, now);
    this._detectShake(pose, now);
    this._detectForward(hands, now);
  }

  _handleTrackingLost(now) {
    if (this._wasTracking && now - this._lastSeenTime > this.config.trackingLostMs) {
      this._wasTracking = false;
      this.dispatchEvent(new CustomEvent("lost"));
    }
  }

  // Simple exponential smoothing per landmark to cut jitter before it
  // reaches gesture logic. Good enough here; swap for a One-Euro filter
  // later if fast gestures ever feel laggy.
  _smooth(key, landmarks) {
    const alpha = this.config.smoothingAlpha;
    const prev = this._smoothed[key];
    if (!prev) {
      this._smoothed[key] = landmarks.map((p) => ({ ...p }));
      return this._smoothed[key];
    }
    const out = landmarks.map((p, i) => ({
      x: lerp(prev[i].x, p.x, alpha),
      y: lerp(prev[i].y, p.y, alpha),
      z: lerp(prev[i].z, p.z, alpha),
      visibility: p.visibility,
    }));
    this._smoothed[key] = out;
    return out;
  }

  emitGesture(type, data = {}) {
    this.dispatchEvent(new CustomEvent("gesture", { detail: { type, ...data, t: performance.now() } }));
  }

  // ---- Gesture detectors ----------------------------------------------

  _detectLean(pose, now) {
    const ls = pose[this.SIDE.L_SHOULDER];
    const rs = pose[this.SIDE.R_SHOULDER];
    const shoulderMidX = (ls.x + rs.x) / 2;

    let offset = (shoulderMidX - this.baseline.shoulderMidX) / this.baseline.shoulderWidth;
    // Display is mirrored (see CONFIG.mirrored), so flip the sign to match
    // what the player visually sees as "their" left/right on screen.
    if (this.config.mirrored) offset = -offset;

    const c = this._lean;
    const enterDir =
      offset < -this.config.leanEnterThreshold ? "left"
      : offset > this.config.leanEnterThreshold ? "right"
      : null;

    if (enterDir && enterDir !== c.state) {
      if (c.pending !== enterDir) {
        c.pending = enterDir;
        c.pendingSince = now;
      } else if (now - c.pendingSince >= this.config.leanHoldMs) {
        c.state = enterDir;
        c.pending = null;
        this.emitGesture("lean", { direction: enterDir });
      }
    } else if (!enterDir && Math.abs(offset) < this.config.leanExitThreshold) {
      c.pending = null;
      if (c.state !== "center") {
        c.state = "center";
        this.emitGesture("lean", { direction: "center" });
      }
    }
  }

  _detectRaise(pose, now) {
    for (const side of ["left", "right"]) {
      const prefix = side === "left" ? "L" : "R";
      const shoulder = pose[this.SIDE[`${prefix}_SHOULDER`]];
      const hip = pose[this.SIDE[`${prefix}_HIP`]];
      const wrist = pose[this.SIDE[`${prefix}_WRIST`]];
      const torsoLength = Math.max(0.05, Math.abs(hip.y - shoulder.y));
      const raisedAmount = (shoulder.y - wrist.y) / torsoLength; // + = wrist above shoulder

      const st = this._raise[side];
      const isRaised = raisedAmount > this.config.raiseThreshold && wrist.visibility > 0.4;

      if (isRaised && !st.raised) {
        if (!st.since) st.since = now;
        if (
          now - st.since >= this.config.raiseHoldMs &&
          now - st.last > this.config.raiseCooldownMs
        ) {
          st.raised = true;
          st.last = now;
          this.emitGesture("raiseHand", { side });
        }
      } else if (!isRaised) {
        st.raised = false;
        st.since = 0;
      }
    }
  }

  _detectClap(pose, now) {
    const lw = pose[this.SIDE.L_WRIST];
    const rw = pose[this.SIDE.R_WRIST];
    const ls = pose[this.SIDE.L_SHOULDER];
    const rs = pose[this.SIDE.R_SHOULDER];
    const shoulderWidth = Math.max(0.05, dist(ls, rs));
    const wristDist = dist(lw, rw) / shoulderWidth;

    const c = this._clap;
    if (wristDist > this.config.clapApartThreshold) c.wasApart = true;

    if (
      c.wasApart &&
      wristDist < this.config.clapDistanceThreshold &&
      now - c.last > this.config.clapCooldownMs
    ) {
      c.wasApart = false;
      c.last = now;
      this.emitGesture("clap");
    }
  }

  _detectShake(pose, now) {
    const ls = pose[this.SIDE.L_SHOULDER];
    const rs = pose[this.SIDE.R_SHOULDER];
    const shoulderWidth = Math.max(0.05, Math.abs(ls.x - rs.x));
    const midX = (ls.x + rs.x) / 2;

    const buf = this._shake.buffer;
    buf.push({ x: midX, t: now });
    while (buf.length && now - buf[0].t > this.config.shakeWindowMs) buf.shift();
    if (buf.length < 5) return;

    let reversals = 0;
    let dir = 0;
    let minX = buf[0].x, maxX = buf[0].x;
    for (let i = 1; i < buf.length; i++) {
      minX = Math.min(minX, buf[i].x);
      maxX = Math.max(maxX, buf[i].x);
      const d = buf[i].x - buf[i - 1].x;
      if (Math.abs(d) < 1e-4) continue;
      const s = Math.sign(d);
      if (dir !== 0 && s !== dir) reversals++;
      dir = s;
    }
    const amplitude = (maxX - minX) / shoulderWidth;

    if (
      reversals >= this.config.shakeMinReversals &&
      amplitude > this.config.shakeAmplitude &&
      now - this._shake.last > this.config.shakeCooldownMs
    ) {
      this._shake.last = now;
      buf.length = 0;
      this.emitGesture("shake");
    }
  }

  // Depth proxy: a monocular webcam can't measure true distance, so we infer
  // "hands pushed toward camera" from hands appearing larger than baseline.
  // This is the least precise detector — expect to retune forwardScaleFactor,
  // or swap the gesture for something more reliable (e.g. "spread hands wide")
  // if it feels flaky during playtesting.
  _detectForward(hands, now) {
    if (!this.baseline.handSize || hands.length < 1) return;
    const avgSize = hands.reduce((sum, h) => sum + bboxDiagonal(h), 0) / hands.length;
    const scale = avgSize / this.baseline.handSize;

    if (
      scale > this.config.forwardScaleFactor &&
      now - this._forward.last > this.config.forwardCooldownMs
    ) {
      this._forward.last = now;
      this.emitGesture("handsForward");
    }
  }
}