// =============================================================================
// bodytracking.js
// -----------------------------------------------------------------------------
// Wraps MediaPipe Tasks-Vision (PoseLandmarker + HandLandmarker) into a single
// BodyTracker class that turns raw webcam landmarks into discrete, game-ready
// gesture EVENTS. This file knows nothing about soda, customers, or scoring —
// game.js consumes it purely through events.
//
// v2 changes (tuned for a fast, "dance-like" pace + easier debugging):
//   - Every hold-time/cooldown/threshold was shortened so gestures register
//     in a few hundred ms instead of half a second-plus. See CONFIG.
//   - HandLandmarker now runs every CONFIG.handFrameSkip-th frame instead of
//     every frame (PoseLandmarker still runs every frame) — pose drives the
//     time-critical gestures (lean/raise/clap/shake), so this cuts total
//     inference cost with the least gameplay impact.
//   - Calibration no longer silently disables 'handsForward' for the whole
//     session if hands weren't visible during calibration; it falls back to
//     a shoulder-width-based estimate instead.
//   - Every frame now reports live diagnostic numbers (current value vs.
//     threshold) via the 'pose' event's `debug` field, so you can see
//     exactly why a gesture did or didn't fire instead of guessing.
//
// v3 changes (from real playtesting):
//   - invertHandedness was flipped to false. Testing showed MediaPipe's raw
//     left/right labels already matched the player's real anatomy for this
//     raw-frame + CSS-mirror setup; the old default was inverting a label
//     that didn't need it, which meant "raise right hand" only ever matched
//     when the (mislabeled) left hand went up too — i.e. you needed both.
//   - _detectShake no longer looks only at shoulder side-to-side sway. It
//     now sums movement across shoulders, hips, AND wrists over a rolling
//     window, so real dancing (hips + hands moving) registers, not just one
//     specific oscillation pattern.
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
//   'pose'        -> fires every processed frame.
//                    detail = { pose: Landmark[33], hands: Landmark[21][], handedness, debug }
//                    `debug` holds live numbers (current value vs. threshold)
//                    for every detector.
//   'gesture'     -> a discrete action was detected. detail.type is one of:
//                      'lean'          detail = { direction: 'left'|'right'|'center' }
//                      'raiseHand'     detail = { side: 'left'|'right' }
//                      'clap'          detail = {}
//                      'shake'         detail = {}
//                      'handsForward'  detail = {}
//
// A NOTE ON LEFT/RIGHT
// -----------------------------------------------------------------------------
// We feed MediaPipe the RAW (unmirrored) camera frame — only the on-screen
// <video> is flipped, via CSS, for a natural "mirror" look. In theory a raw
// (non-selfie-oriented) frame should come out with left/right reversed
// relative to the player, which is why invertHandedness existed — but real
// testing showed MediaPipe's raw labels already matched the player's actual
// anatomy here, so the correction is now OFF by default. If gestures ever
// feel mirrored again (e.g. "raise right" only fires when you raise your
// left), that's the flag to toggle back to true.
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
export const CONFIG = {
  targetFps: 34,
  handFrameSkip: 2, // run HandLandmarker on 1 out of every N processed frames
  smoothingAlpha: 0.65, // 0..1, higher = snappier but jitterier

  leanEnterThreshold: 0.26,
  leanExitThreshold: 0.13,
  leanHoldMs: 70,

  raiseThreshold: 0.28,
  raiseHoldMs: 50,
  raiseCooldownMs: 220,

  clapApartThreshold: 0.8,
  clapDistanceThreshold: 0.4,
  clapCooldownMs: 250,

  shakeWindowMs: 500,
  shakeEnergyThreshold: 4.5, // sum of normalized movement across shoulders/hips/wrists in the window
  shakeCooldownMs: 350,

  forwardScaleFactor: 1.35,
  forwardCooldownMs: 280,

  trackingLostMs: 800,

  mirrored: true,
  invertHandedness: false,
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
    this._frameCount = 0;
    this._cachedHands = [];
    this._cachedHandedness = [];

    this._debug = {
      calibrated: false,
      leanOffset: 0, leanThreshold: this.config.leanEnterThreshold,
      raiseLeft: 0, raiseRight: 0, raiseThreshold: this.config.raiseThreshold,
      clapDist: 0, clapThreshold: this.config.clapDistanceThreshold,
      shakeEnergy: 0, shakeThreshold: this.config.shakeEnergyThreshold,
      forwardScale: 0, forwardThreshold: this.config.forwardScaleFactor,
    };

    this._lean = { state: "center", pending: null, pendingSince: 0 };
    this._raise = {
      left: { raised: false, since: 0, last: 0 },
      right: { raised: false, since: 0, last: 0 },
    };
    this._clap = { wasApart: true, last: 0 };
    this._shake = { buffer: [], last: 0, prevPoints: null };
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

  // Requires BOTH a minimum time AND a minimum number of frames where a body
  // was actually detected before finalizing a baseline. Without this guard,
  // if the models are still warming up or you're not in frame yet, it would
  // happily average zero real samples into a baseline of (0, 0) — after
  // which every real reading looks like a giant, permanent lean in one
  // direction, and leaning further that way (or the other way) does nothing.
  calibrate(minDurationMs = 1200, { minSamples = 15, maxWaitMs = 6000 } = {}) {
    return new Promise((resolve, reject) => {
      const samples = [];
      const startedAt = performance.now();
      const collect = () => {
        if (this._lastFrame) samples.push(this._lastFrame);
        const elapsed = performance.now() - startedAt;
        const haveEnough = samples.length >= minSamples && elapsed >= minDurationMs;

        if (haveEnough) {
          this.baseline = this._computeBaseline(samples);
          this._debug.calibrated = true;
          this.dispatchEvent(new CustomEvent("calibrated", { detail: this.baseline }));
          resolve(this.baseline);
        } else if (elapsed >= maxWaitMs) {
          reject(
            new Error(
              "Couldn't get a clear view of you. Step back so your shoulders and hands are both visible, check the lighting, then try again."
            )
          );
        } else {
          requestAnimationFrame(collect);
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

    const avgShoulderWidth = Math.max(0.05, shoulderWidth / n);
    return {
      shoulderMidX: shoulderMidX / n,
      shoulderWidth: avgShoulderWidth,
      // Falls back to a shoulder-width estimate if hands weren't visible
      // during calibration, so 'handsForward' doesn't get silently disabled
      // for the whole session.
      handSize: handCount ? handSize / handCount : avgShoulderWidth * 0.35,
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

    this._frameCount++;
    let hands = this._cachedHands;
    let handedness = this._cachedHandedness;
    if (this._frameCount % this.config.handFrameSkip === 0) {
      const handResult = this.hands.detectForVideo(this.video, now);
      hands = (handResult.landmarks || []).map((lm, i) => this._smooth(`hand${i}`, lm));
      handedness = (handResult.handedness || []).map((cats) =>
        cats.map((c) => ({
          ...c,
          categoryName:
            this.config.invertHandedness
              ? c.categoryName === "Left" ? "Right" : "Left"
              : c.categoryName,
        }))
      );
      this._cachedHands = hands;
      this._cachedHandedness = handedness;
    }

    this._lastFrame = { pose, hands, handedness };

    if (this.baseline) {
      this._detectLean(pose, now);
      this._detectRaise(pose, now);
      this._detectClap(pose, now);
      this._detectShake(pose, now);
      this._detectForward(hands, now);
    }

    this.dispatchEvent(
      new CustomEvent("pose", { detail: { ...this._lastFrame, debug: { ...this._debug } } })
    );
  }

  _handleTrackingLost(now) {
    if (this._wasTracking && now - this._lastSeenTime > this.config.trackingLostMs) {
      this._wasTracking = false;
      this.dispatchEvent(new CustomEvent("lost"));
    }
  }

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
    if (this.config.mirrored) offset = -offset;
    this._debug.leanOffset = offset;

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
      const raisedAmount = (shoulder.y - wrist.y) / torsoLength;
      this._debug[side === "left" ? "raiseLeft" : "raiseRight"] = raisedAmount;

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
    this._debug.clapDist = wristDist;

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

  // Sums how much shoulders, hips, and wrists moved frame-to-frame, over a
  // rolling time window. This deliberately doesn't care about direction or
  // pattern — swaying hips, waving hands, bouncing shoulders, or any mix of
  // the three all add up to "shake". That matches actual dancing much better
  // than requiring one specific side-to-side oscillation.
  _detectShake(pose, now) {
    const points = [
      pose[this.SIDE.L_SHOULDER], pose[this.SIDE.R_SHOULDER],
      pose[this.SIDE.L_HIP], pose[this.SIDE.R_HIP],
      pose[this.SIDE.L_WRIST], pose[this.SIDE.R_WRIST],
    ];
    const shoulderWidth = Math.max(0.05, dist(pose[this.SIDE.L_SHOULDER], pose[this.SIDE.R_SHOULDER]));

    let frameEnergy = 0;
    const prev = this._shake.prevPoints;
    if (prev) {
      for (let i = 0; i < points.length; i++) frameEnergy += dist(points[i], prev[i]);
    }
    this._shake.prevPoints = points.map((p) => ({ x: p.x, y: p.y }));

    const buf = this._shake.buffer;
    buf.push({ e: frameEnergy / shoulderWidth, t: now });
    while (buf.length && now - buf[0].t > this.config.shakeWindowMs) buf.shift();

    const totalEnergy = buf.reduce((sum, b) => sum + b.e, 0);
    this._debug.shakeEnergy = totalEnergy;

    if (
      totalEnergy > this.config.shakeEnergyThreshold &&
      now - this._shake.last > this.config.shakeCooldownMs
    ) {
      this._shake.last = now;
      buf.length = 0;
      this.emitGesture("shake");
    }
  }

  _detectForward(hands, now) {
    if (!this.baseline.handSize || hands.length < 1) return;
    const avgSize = hands.reduce((sum, h) => sum + bboxDiagonal(h), 0) / hands.length;
    const scale = avgSize / this.baseline.handSize;
    this._debug.forwardScale = scale;

    if (
      scale > this.config.forwardScaleFactor &&
      now - this._forward.last > this.config.forwardCooldownMs
    ) {
      this._forward.last = now;
      this.emitGesture("handsForward");
    }
  }
}
