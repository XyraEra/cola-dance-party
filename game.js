// =============================================================================
// game.js
// -----------------------------------------------------------------------------
// The soda-machine game brain. Consumes BodyTracker purely through its event
// API (see bodytracking.js header) — this file never touches MediaPipe
// directly. It owns: recipes, order generation, step matching, scoring,
// lives/levels, and all DOM updates.
//
// REQUIRED DOM ELEMENTS (index.html must provide all of these by id)
// -----------------------------------------------------------------------------
//   webcam                 <video>   the live camera feed, passed to BodyTracker
//   skeleton-canvas         <canvas>  OPTIONAL debug overlay — same size/position
//                                     as #webcam, same mirror transform, or omit
//                                     it entirely and skeleton drawing is skipped
//   start-screen            <div>     initial landing screen
//   start-button            <button>  begins camera init + calibration
//   calibration-overlay     <div>     shown while calibrating, starts hidden
//   calibration-message     <*>       text updated with countdown / errors
//   hud                     <div>     main in-round UI, starts hidden
//   score-value             <*>       current score
//   combo-value             <*>       current combo streak
//   level-value             <*>       current level
//   lives-value             <*>       remaining lives (hearts)
//   timer-bar-fill          <div>     width is animated 100%->0% per order
//   customer-name           <*>       flavor text: which customer is ordering
//   order-name              <*>       the drink name, e.g. "Cherry Cola"
//   steps-list              <ul/ol>   populated with one <li class="step"> per
//                                     ingredient, each needs data-index
//   current-action-icon     <*>       big icon for the step the player needs now
//   current-action-label    <*>       matching label, e.g. "Raise Right Hand"
//   feedback                <div>     star-rating / "Missed!" popup, starts hidden
//   tracking-lost-banner    <div>     "step into frame" warning, starts hidden
//   game-over-screen        <div>     starts hidden
//   final-score             <*>       shown on game over
//   restart-button          <button>  returns to a fresh round
//
// CSS classes this file toggles: "hidden", "show", "pending", "current",
// "done", "error-flash", "success", "missed", "level-up".
// =============================================================================

import { BodyTracker } from "./bodytracking.js";

// ---- Game-balance tuning ----------------------------------------------------
const CONFIG = {
  startingLives: 3,
  calibrationMs: 1500,
  orderIntroMs: 1200, // pause to read the order before the timer starts
  resultDisplayMs: 1400, // how long the star popup / miss popup stays up
  levelUpEveryOrders: 3,
  baseTimeMs: 9000,
  perStepTimeMs: 2400,
  perLevelTimePenaltyMs: 500,
  minTimeMs: 6000,
  maxTimeMs: 26000,
  comboScoreBonus: 0.12, // +12% score per combo stack beyond the first
  showSkeleton: true,
};

// ---- Ingredient catalog -----------------------------------------------------
// Each entry maps one recipe "beat" to a physical gesture + variant. Recipes
// below just chain these in different orders/combinations — that's what gives
// menu variety without needing more than 5 physical actions.
const INGREDIENTS = {
  colaBase: { gesture: "lean", match: { direction: "left" }, label: "Lean Left — Cola", icon: "🥤" },
  coffeeBase: { gesture: "lean", match: { direction: "right" }, label: "Lean Right — Coffee", icon: "☕" },
  pour: { gesture: "raiseHand", match: { side: "right" }, label: "Raise Right Hand — Pour", icon: "🙌" },
  espresso: { gesture: "raiseHand", match: { side: "right" }, label: "Raise Right Hand — Espresso Shot", icon: "⚡" },
  ice: { gesture: "clap", match: {}, label: "Clap — Add Ice", icon: "🧊" },
  cherry: { gesture: "clap", match: {}, label: "Clap — Add Cherry Syrup", icon: "🍒" },
  mix: { gesture: "shake", match: {}, label: "Shake — Mix", icon: "🕺" },
  serve: { gesture: "handsForward", match: {}, label: "Hands Forward — Serve", icon: "✋" },
};

// ---- Recipe book -------------------------------------------------------------
// minLevel gates when a recipe can appear. Longer/harder recipes unlock as
// the player levels up, which is how "more ingredients" is implemented.
const RECIPE_BOOK = [
  { name: "Black Coffee", minLevel: 1, ingredients: ["coffeeBase", "pour", "mix", "serve"] },
  { name: "Cola with Ice", minLevel: 1, ingredients: ["colaBase", "pour", "ice", "mix", "serve"] },
  { name: "Coffee + Cola", minLevel: 1, ingredients: ["coffeeBase", "colaBase", "pour", "mix", "serve"] },
  { name: "Cherry Cola", minLevel: 2, ingredients: ["colaBase", "pour", "cherry", "mix", "serve"] },
  { name: "Iced Coffee", minLevel: 2, ingredients: ["coffeeBase", "pour", "ice", "mix", "serve"] },
  { name: "Double Espresso Cola", minLevel: 3, ingredients: ["colaBase", "pour", "espresso", "espresso", "mix", "serve"] },
  { name: "Cherry Ice Cola", minLevel: 3, ingredients: ["colaBase", "pour", "cherry", "ice", "mix", "serve"] },
  { name: "The Works", minLevel: 4, ingredients: ["colaBase", "coffeeBase", "pour", "espresso", "cherry", "ice", "mix", "serve"] },
];

const CUSTOMER_NAMES = [
  "Trucker Joe", "Roller-Skate Rita", "Neon Nina", "Biker Bill", "Jitterbug Jill",
  "Sax Man Sam", "Malt Shop Mia", "Hot Rod Hank", "Greaser Gus", "Diner Dot",
];

const STAR_MULT = { 1: 0.5, 2: 0.75, 3: 1 };

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}
function pickRandom(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

// ---- DOM references ----------------------------------------------------------
// This script is loaded with type="module", which behaves like `defer`, so
// the document is already parsed by the time this runs — no need to wait
// for DOMContentLoaded.
const videoEl = document.getElementById("webcam");
const skeletonCanvas = document.getElementById("skeleton-canvas");
const skeletonCtx = skeletonCanvas ? skeletonCanvas.getContext("2d") : null;

const startScreenEl = document.getElementById("start-screen");
const startButtonEl = document.getElementById("start-button");
const calibrationOverlayEl = document.getElementById("calibration-overlay");
const calibrationMessageEl = document.getElementById("calibration-message");

const hudEl = document.getElementById("hud");
const scoreEl = document.getElementById("score-value");
const comboEl = document.getElementById("combo-value");
const levelEl = document.getElementById("level-value");
const livesEl = document.getElementById("lives-value");
const timerBarEl = document.getElementById("timer-bar-fill");
const customerNameEl = document.getElementById("customer-name");
const orderNameEl = document.getElementById("order-name");
const stepsListEl = document.getElementById("steps-list");
const currentActionIconEl = document.getElementById("current-action-icon");
const currentActionLabelEl = document.getElementById("current-action-label");
const feedbackEl = document.getElementById("feedback");
const trackingLostBannerEl = document.getElementById("tracking-lost-banner");

const gameOverScreenEl = document.getElementById("game-over-screen");
const finalScoreEl = document.getElementById("final-score");
const restartButtonEl = document.getElementById("restart-button");

// ---- Tracker instance ---------------------------------------------------------
const tracker = new BodyTracker(videoEl);

// ---- Mutable game state ---------------------------------------------------------
let state = "MENU"; // MENU | CALIBRATING | INTRO | PLAYING | PAUSED | RESULT | GAME_OVER
let score = 0;
let combo = 0;
let level = 1;
let lives = CONFIG.startingLives;
let ordersCompleted = 0;
let lastRecipeName = null;

let currentRecipe = null;
let currentStepIndex = 0;
let mistakes = 0;
let roundDuration = 0;
let roundDeadline = 0;
let pausedRemainingMs = null;

// =============================================================================
// Recipe selection
// =============================================================================
function pickRecipe() {
  const pool = RECIPE_BOOK.filter((r) => r.minLevel <= level);
  // Bias toward recipes closer to (or at) the player's current level, so
  // difficulty ramps up without losing variety from easier ones.
  const weighted = pool.flatMap((r) => Array(clamp(level - r.minLevel + 1, 1, 3)).fill(r));

  let choice;
  let attempts = 0;
  do {
    choice = pickRandom(weighted);
    attempts++;
  } while (choice.name === lastRecipeName && attempts < 5);

  lastRecipeName = choice.name;
  return choice;
}

function computeRoundDuration(recipe) {
  const raw =
    CONFIG.baseTimeMs +
    recipe.ingredients.length * CONFIG.perStepTimeMs -
    (level - 1) * CONFIG.perLevelTimePenaltyMs;
  return clamp(raw, CONFIG.minTimeMs, CONFIG.maxTimeMs);
}

// =============================================================================
// Step matching
// =============================================================================
// Returns true (correct match), false (wrong variant — counts as a mistake),
// or null (unrelated gesture — ignored, no penalty).
function stepMatches(ingredientKey, gestureDetail) {
  const ing = INGREDIENTS[ingredientKey];
  if (ing.gesture !== gestureDetail.type) return null;
  for (const key in ing.match) {
    if (ing.match[key] !== gestureDetail[key]) return false;
  }
  return true;
}

function onGesture(e) {
  if (state !== "PLAYING") return;
  const detail = e.detail;

  // Returning to a neutral lean is a natural resting transition, not a
  // deliberate selection — never treat it as a wrong answer.
  if (detail.type === "lean" && detail.direction === "center") return;

  const expectedKey = currentRecipe.ingredients[currentStepIndex];
  const result = stepMatches(expectedKey, detail);

  if (result === true) advanceStep();
  else if (result === false) registerMistake();
}

function advanceStep() {
  markStepComplete(currentStepIndex);
  currentStepIndex++;
  if (currentStepIndex >= currentRecipe.ingredients.length) {
    succeedRound();
  } else {
    highlightCurrentStep();
  }
}

function registerMistake() {
  mistakes++;
  flashCurrentStepError();
}

// =============================================================================
// Round lifecycle
// =============================================================================
function nextCustomer() {
  if (state === "GAME_OVER") return;

  currentRecipe = pickRecipe();
  currentStepIndex = 0;
  mistakes = 0;

  customerNameEl.textContent = pickRandom(CUSTOMER_NAMES);
  orderNameEl.textContent = currentRecipe.name;
  renderSteps(currentRecipe);
  highlightCurrentStep();
  timerBarEl.style.width = "100%";

  state = "INTRO";
  setTimeout(() => {
    if (state !== "INTRO") return; // e.g. tracking was lost mid-intro
    roundDuration = computeRoundDuration(currentRecipe);
    roundDeadline = performance.now() + roundDuration;
    state = "PLAYING";
  }, CONFIG.orderIntroMs);
}

function computeStars(mistakeCount, timeRemainingFraction) {
  if (mistakeCount === 0 && timeRemainingFraction > 0.4) return 3;
  if (mistakeCount <= 1 && timeRemainingFraction > 0.15) return 2;
  return 1;
}

function scoreForOrder(recipe, stars, comboCount) {
  const base = 100 + recipe.ingredients.length * 25;
  const comboMult = 1 + (comboCount - 1) * CONFIG.comboScoreBonus;
  return Math.round(base * STAR_MULT[stars] * comboMult);
}

function succeedRound() {
  state = "RESULT";
  const timeRemainingFraction = Math.max(0, (roundDeadline - performance.now()) / roundDuration);
  const stars = computeStars(mistakes, timeRemainingFraction);

  combo++;
  ordersCompleted++;
  score += scoreForOrder(currentRecipe, stars, combo);
  if (ordersCompleted % CONFIG.levelUpEveryOrders === 0) levelUp();

  updateHUD();
  showResult({ stars, missed: false });
  setTimeout(nextCustomer, CONFIG.resultDisplayMs);
}

function failRound() {
  state = "RESULT";
  combo = 0;
  lives = Math.max(0, lives - 1);

  updateHUD();
  showResult({ stars: 0, missed: true });
  setTimeout(() => (lives <= 0 ? gameOver() : nextCustomer()), CONFIG.resultDisplayMs);
}

function levelUp() {
  level++;
  levelEl.classList.add("level-up");
  setTimeout(() => levelEl.classList.remove("level-up"), 600);
}

function gameOver() {
  state = "GAME_OVER";
  finalScoreEl.textContent = score;
  gameOverScreenEl.classList.remove("hidden");
  hudEl.classList.add("hidden");
}

function beginGame() {
  score = 0;
  combo = 0;
  level = 1;
  lives = CONFIG.startingLives;
  ordersCompleted = 0;
  lastRecipeName = null;
  updateHUD();
  gameOverScreenEl.classList.add("hidden");
  hudEl.classList.remove("hidden");
  nextCustomer();
}

// =============================================================================
// Rendering
// =============================================================================
function renderSteps(recipe) {
  stepsListEl.innerHTML = "";
  recipe.ingredients.forEach((key, i) => {
    const ing = INGREDIENTS[key];
    const li = document.createElement("li");
    li.className = "step pending";
    li.dataset.index = String(i);
    li.innerHTML = `<span class="step-icon">${ing.icon}</span><span class="step-label">${ing.label}</span>`;
    stepsListEl.appendChild(li);
  });
}

function markStepComplete(index) {
  const li = stepsListEl.querySelector(`[data-index="${index}"]`);
  if (li) {
    li.classList.remove("pending", "current");
    li.classList.add("done");
  }
}

function highlightCurrentStep() {
  stepsListEl.querySelectorAll(".step").forEach((li) => li.classList.remove("current"));
  const li = stepsListEl.querySelector(`[data-index="${currentStepIndex}"]`);
  if (li) li.classList.add("current");

  const key = currentRecipe.ingredients[currentStepIndex];
  const ing = key ? INGREDIENTS[key] : null;
  currentActionIconEl.textContent = ing ? ing.icon : "";
  currentActionLabelEl.textContent = ing ? ing.label : "";
}

function flashCurrentStepError() {
  const li = stepsListEl.querySelector(".current");
  if (!li) return;
  li.classList.add("error-flash");
  setTimeout(() => li.classList.remove("error-flash"), 350);
}

function showResult({ stars, missed }) {
  feedbackEl.classList.remove("success", "missed");
  feedbackEl.classList.add("show", missed ? "missed" : "success");
  feedbackEl.innerHTML = missed
    ? `<div class="feedback-title">Missed!</div>`
    : `<div class="feedback-stars">${"⭐".repeat(stars)}${"☆".repeat(3 - stars)}</div>
       <div class="feedback-title">${stars === 3 ? "Perfect!" : stars === 2 ? "Nice!" : "Served!"}</div>`;
  setTimeout(() => feedbackEl.classList.remove("show"), CONFIG.resultDisplayMs - 200);
}

function updateHUD() {
  scoreEl.textContent = String(score);
  comboEl.textContent = combo > 1 ? `x${combo}` : "";
  levelEl.textContent = String(level);
  livesEl.textContent = "❤️".repeat(lives) + "🖤".repeat(CONFIG.startingLives - lives);
}

function showTrackingBanner(show) {
  trackingLostBannerEl.classList.toggle("hidden", !show);
}

function drawSkeleton({ pose, hands }) {
  if (!CONFIG.showSkeleton || !skeletonCanvas) return;
  if (videoEl.videoWidth && skeletonCanvas.width !== videoEl.videoWidth) {
    skeletonCanvas.width = videoEl.videoWidth;
    skeletonCanvas.height = videoEl.videoHeight;
  }
  const { width, height } = skeletonCanvas;
  skeletonCtx.clearRect(0, 0, width, height);

  skeletonCtx.fillStyle = "#39ff88";
  for (const p of pose) {
    if (p.visibility != null && p.visibility < 0.4) continue;
    skeletonCtx.beginPath();
    skeletonCtx.arc(p.x * width, p.y * height, 4, 0, Math.PI * 2);
    skeletonCtx.fill();
  }

  skeletonCtx.fillStyle = "#ffd23f";
  for (const hand of hands) {
    for (const p of hand) {
      skeletonCtx.beginPath();
      skeletonCtx.arc(p.x * width, p.y * height, 3, 0, Math.PI * 2);
      skeletonCtx.fill();
    }
  }
}

// =============================================================================
// Calibration flow
// =============================================================================
function runCalibrationCountdown(ms) {
  return new Promise((resolve) => {
    const start = performance.now();
    function tick() {
      const elapsed = performance.now() - start;
      const remaining = Math.ceil((ms - elapsed) / 1000);
      calibrationMessageEl.textContent =
        remaining > 0 ? `Stand naturally, hold still... ${remaining}` : "Hold it!";
      if (elapsed < ms) requestAnimationFrame(tick);
      else resolve();
    }
    tick();
  });
}

async function startFlow() {
  startButtonEl.disabled = true;
  startButtonEl.textContent = "Loading camera & AI models...";
  try {
    await tracker.init();
    // Detection loop must be running before calibrate() can sample frames.
    tracker.start();

    startScreenEl.classList.add("hidden");
    calibrationOverlayEl.classList.remove("hidden");
    calibrationMessageEl.textContent = "Stand naturally, hold still...";

    await Promise.all([
      tracker.calibrate(CONFIG.calibrationMs),
      runCalibrationCountdown(CONFIG.calibrationMs),
    ]);

    calibrationOverlayEl.classList.add("hidden");
    beginGame();
  } catch (err) {
    console.error(err);
    calibrationMessageEl.textContent = err.message || "Something went wrong.";
    calibrationOverlayEl.classList.remove("hidden");
    startScreenEl.classList.remove("hidden");
    startButtonEl.disabled = false;
    startButtonEl.textContent = "Try Again";
  }
}

// =============================================================================
// Wiring
// =============================================================================
startButtonEl.addEventListener("click", startFlow);
restartButtonEl.addEventListener("click", beginGame);

tracker.addEventListener("gesture", onGesture);

tracker.addEventListener("pose", (e) => drawSkeleton(e.detail));

tracker.addEventListener("lost", () => {
  showTrackingBanner(true);
  if (state === "PLAYING") {
    pausedRemainingMs = roundDeadline - performance.now();
    state = "PAUSED";
  }
});

tracker.addEventListener("found", () => {
  showTrackingBanner(false);
  if (state === "PAUSED") {
    roundDeadline = performance.now() + pausedRemainingMs;
    state = "PLAYING";
  }
});

// Central UI tick: animates the timer bar and triggers timeouts. Kept
// separate from BodyTracker's internal detection loop on purpose — this
// one only touches game state, never MediaPipe.
function gameTick(now) {
  if (state === "PLAYING") {
    const remaining = Math.max(0, roundDeadline - now);
    timerBarEl.style.width = `${(remaining / roundDuration) * 100}%`;
    if (remaining <= 0) failRound();
  }
  requestAnimationFrame(gameTick);
}
requestAnimationFrame(gameTick);