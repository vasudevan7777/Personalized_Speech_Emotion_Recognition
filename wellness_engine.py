import random
import numpy as np
import librosa

BASE_EMOTION_RISKS = {
    'calm': 10, 'neutral': 15, 'happy': 20, 'surprised': 35,
    'sad': 55,  'disgust': 62, 'fearful': 78, 'angry': 88,
}

DISCLAIMER_TEXT = (
    "⚠️ **Disclaimer**: This is an informational AI tool for voice analysis. It does "
    "not constitute a medical evaluation, psychiatric assessment, or clinical diagnosis. "
    "If you or someone you know is experiencing emotional distress or a mental health crisis, "
    "please reach out to a professional healthcare provider or contact your local crisis hotline."
)


def _analyze_acoustics(y: np.ndarray, sr: int = 22050) -> dict:
    if y is None or len(y) == 0:
        return {"pitch_std": 0.0, "rms_mean": 0.0, "rms_std": 0.0, "intensity_bucket": "unknown"}

    pitches, _ = librosa.piptrack(y=y, sr=sr)
    voiced = pitches[pitches > 0]
    pitch_std = float(np.std(voiced)) if len(voiced) > 0 else 0.0

    rms_frames = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms_frames)) if len(rms_frames) > 0 else 0.0
    rms_std  = float(np.std(rms_frames))  if len(rms_frames) > 0 else 0.0

    if rms_mean < 0.02:
        bucket = "whisper"
    elif rms_mean < 0.12:
        bucket = "normal"
    else:
        bucket = "loud"

    return {
        "pitch_std":        round(pitch_std, 3),
        "rms_mean":         round(rms_mean,  4),
        "rms_std":          round(rms_std,   4),
        "intensity_bucket": bucket,
    }


def analyze_acoustic_arousal(audio_data, sr: int = 22050):
    """Backward-compatible wrapper."""
    ac = _analyze_acoustics(audio_data, sr)
    return ac["pitch_std"], ac["rms_std"], ac["rms_mean"]


def calculate_emotional_risk(emotion: str, confidence: float, audio_data=None, sr: int = 22050) -> dict:
    """Calculate Emotional Risk Score (0-100) with explainable components."""
    emotion   = emotion.lower()
    base_risk = BASE_EMOTION_RISKS.get(emotion, 25)
    baseline  = 35.0

    w = confidence ** 2
    conf_adjusted = w * base_risk + (1.0 - w) * baseline

    components = [
        ("Base emotion score",   round(base_risk, 1)),
        ("Confidence weighting", round(conf_adjusted - base_risk, 1)),
    ]
    adjustment = 0.0

    if audio_data is not None and len(np.asarray(audio_data)) > 0:
        ac = _analyze_acoustics(np.asarray(audio_data, dtype=np.float32), sr)
        pitch_std = ac["pitch_std"]
        rms_std   = ac["rms_std"]
        bucket    = ac["intensity_bucket"]

        HIGH_AROUSAL = {'angry', 'fearful', 'surprised'}
        LOW_ENERGY   = {'sad', 'neutral'}
        DISTRESS     = {'sad', 'fearful', 'disgust'}

        pitch_adj = 0.0
        if pitch_std > 120:
            pitch_adj = +12.0
        elif pitch_std > 80:
            pitch_adj = +8.0 if emotion in HIGH_AROUSAL else +4.0
        elif 0 < pitch_std < 15 and emotion in LOW_ENERGY:
            pitch_adj = +8.0
        elif 15 <= pitch_std < 40 and emotion in LOW_ENERGY:
            pitch_adj = +4.0
        if abs(pitch_adj) > 0:
            components.append((f"Pitch variability (std={pitch_std:.0f})", round(pitch_adj, 1)))
        adjustment += pitch_adj

        intensity_adj = 0.0
        if bucket == "loud" and emotion in HIGH_AROUSAL:
            intensity_adj = +8.0
            components.append(("Loud + high-arousal", intensity_adj))
        elif bucket == "whisper" and emotion in DISTRESS:
            intensity_adj = +6.0
            components.append(("Whisper + distress", intensity_adj))
        elif bucket == "whisper" and emotion in {'calm', 'happy', 'neutral'}:
            intensity_adj = -3.0
            components.append(("Whisper + positive", intensity_adj))
        adjustment += intensity_adj

        vol_adj = 0.0
        if rms_std > 0.05 and emotion in {'angry', 'fearful'}:
            vol_adj = +5.0
            components.append((f"High volume fluctuation (rms_std={rms_std:.3f})", vol_adj))
        adjustment += vol_adj

    final_score = float(np.clip(conf_adjusted + adjustment, 0.0, 100.0))

    if final_score < 30:
        category = "Low"
    elif final_score < 60:
        category = "Moderate"
    elif final_score < 85:
        category = "High"
    else:
        category = "Extreme"

    return {
        "base_risk":           base_risk,
        "confidence_adjusted": round(conf_adjusted, 1),
        "components":          components,
        "acoustic_adjustments": [(c[0], c[1]) for c in components if c[0] != "Base emotion score"],
        "final_score":         round(final_score, 1),
        "category":            category,
    }


def get_wellness_assessment(emotion: str, risk_score: float) -> dict:
    """Return wellness status label and vocal interpretation for the predicted emotion."""
    emotion = emotion.lower()

    if risk_score < 30:
        if emotion == 'happy':
            status = "Vibrant & Positive"
        elif emotion in ('calm', 'neutral'):
            status = "Balanced & Calm"
        else:
            status = "Slightly Unsettled"
    elif risk_score < 60:
        if emotion == 'happy':
            status = "Expressive / Enthusiastic"
        elif emotion in ('surprised', 'neutral'):
            status = "Receptive / Attentive"
        else:
            status = "Subdued / Vulnerable"
    elif risk_score < 85:
        if emotion in ('angry', 'fearful'):
            status = "Agitated / Stressed"
        elif emotion == 'sad':
            status = "Distressed / Low Energy"
        else:
            status = "Stressed / Elevated Arousal"
    else:
        status = "Overwhelmed / High Distress"

    interpretations = {
        'calm':      "Your voice sounds tranquil and relaxed, reflecting emotional ease and physical comfort.",
        'neutral':   "Your voice has a flat, even tone suggesting objectivity and cognitive control.",
        'happy':     "Your voice displays high tonal range and brightness — a signature of joy and positive energy.",
        'sad':       "Your voice exhibits lower pitch and quieter volume, typical of grief or introspection.",
        'angry':     "Your voice carries high energy and sharp harmonics, indicating frustration or strong dissatisfaction.",
        'fearful':   "Your voice sounds unstable with elevated pitch variation, reflecting anxiety or high arousal.",
        'disgust':   "Your voice shows nasal coloring and low-pitch drops, indicating strong aversion or disapproval.",
        'surprised': "Your voice has sudden pitch jumps and rapid intensity bursts — a startle response to unexpected stimuli.",
    }

    interpretation = interpretations.get(
        emotion,
        f"Your voice is expressing characteristics associated with a {emotion} emotional state."
    )
    return {"status": status, "interpretation": interpretation}


_RECOMMENDATION_BANK = {
    'calm': [
        "Maintain this peaceful state by practicing 2 minutes of focused gratitude.",
        "Use this stable state for deep work, planning, or complex decision-making.",
        "Write down your current mindset to reference during stressful times.",
        "Share your calm energy — a relaxed presence positively influences those around you.",
        "Take a short walk in nature to deepen the sense of ease you are feeling.",
        "Try a brief body scan: close your eyes and release any areas of tension.",
    ],
    'neutral': [
        "You are at a balanced baseline — an excellent time for focused, productive work.",
        "Incorporate a light stretch to keep physical energy aligned with your neutral mind.",
        "Engage in creative brainstorming while cognitive filters are relaxed.",
        "Stay hydrated and maintain your current routine; stability is valuable.",
        "Plan your next goals while your thinking is clear and unbiased.",
        "A short mindful breathing break can deepen this already balanced state.",
    ],
    'happy': [
        "Capture this positive energy — write down 3 things you are happy about right now.",
        "Reach out to a friend, colleague, or loved one to spread the warmth.",
        "Use this high-motivation window to tackle a challenging or long-stalled task.",
        "Continue your positive activities — happiness correlates with sustained productivity.",
        "Celebrate a small win today. Positive reinforcement matters.",
        "Channel this energy into a creative project or something you have been putting off.",
        "Share your good mood in a team setting — positive affect is genuinely contagious.",
    ],
    'sad': [
        "Allow yourself to feel. Low-energy moments are valid — be gentle with yourself.",
        "Try a grounding touch: wrap yourself in a warm blanket or hold a warm cup of tea.",
        "Step outside for 5 minutes of natural light to lift your energy.",
        "Take a short walk — mild movement is one of the most effective mood regulators.",
        "Talk with someone you trust about what you are experiencing.",
        "Listen to calming or uplifting music that resonates with your current mood.",
        "Consider journaling: writing about feelings can reduce their emotional intensity.",
        "Rest if you need to — honour your body's signals rather than pushing through.",
    ],
    'angry': [
        "Try the 4-7-8 breathing method: inhale 4 s, hold 7 s, exhale slowly 8 s.",
        "Release physical tension: open and close your fists, do shoulder shrugs, or walk briskly.",
        "Step away briefly from the triggering environment before responding or deciding.",
        "Drink a full glass of cold water — hydration helps regulate emotional arousal.",
        "Practice deep breathing: 10 slow, deliberate breaths before re-engaging.",
        "Write down what triggered the anger; externalising it reduces its grip.",
        "Wait 10 minutes before sending any message or making any decision.",
        "Resume your work or conversation only after you feel physically calmer.",
    ],
    'fearful': [
        "Ground yourself: name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste.",
        "Affirm your safety: take slow deep breaths, remind yourself you are in a safe space.",
        "Reduce sensory inputs: dim lights, silence notifications, close your eyes briefly.",
        "Slow your breathing to 6 breaths per minute to activate the parasympathetic system.",
        "Focus on one small, concrete task to redirect attention away from the source of fear.",
        "Talk to someone you trust — verbalising fear reduces its perceived intensity.",
        "Try progressive muscle relaxation: tense and release each muscle group from feet to shoulders.",
    ],
    'disgust': [
        "Cleanse your space: tidy your desk, open a window for fresh air, or wash your hands.",
        "Reflect on what caused the aversion and list steps to set better boundaries.",
        "Drink a glass of cold water to physically reset your sensory focus.",
        "Take a short mental reset: switch to a completely different, pleasant task for 5-10 minutes.",
        "Shift your attention deliberately to something neutral or enjoyable.",
        "Name the feeling clearly — labeling emotions reduces their intensity.",
        "Avoid ruminating on the source; redirect attention is more effective than analysis.",
    ],
    'surprised': [
        "Stabilise your heart rate with 3 slow, deep diaphragmatic breaths.",
        "Take a moment to process the surprising event before acting on impulse.",
        "Determine if the surprise requires an immediate response or simply new information to absorb.",
        "Briefly review the facts — separating facts from interpretations helps.",
        "Talk through the surprise with someone if it feels overwhelming.",
        "Give yourself permission to pause — considered responses are always better than reactive ones.",
    ],
}

_HIGH_RISK_PREFIX   = "Elevated arousal detected — pause your current activity and take a 2-minute break."
_EXTREME_RISK_SUFFIX = "Consider talking to a trusted friend or a counseling professional to help process these intense feelings."


def get_recommendations(emotion: str, risk_score: float) -> list:
    """Return 3 randomly sampled recommendations, with escalation messages for high risk."""
    emotion  = emotion.lower()
    bank     = _RECOMMENDATION_BANK.get(emotion, ["Take a few slow, steady breaths to centre yourself."])
    selected = random.sample(bank, min(3, len(bank)))

    if risk_score >= 60:
        selected.insert(0, _HIGH_RISK_PREFIX)
    if risk_score >= 85:
        selected.append(_EXTREME_RISK_SUFFIX)

    return selected


def calculate_stability(history_scores: list):
    """Return (stability_label, emoji) based on risk score standard deviation."""
    if len(history_scores) < 3:
        return "Insufficient data (need 3+ predictions)", "⚪"
    std_dev = np.std(history_scores)
    if std_dev < 10:
        return "Stable",          "🟢"
    elif std_dev < 25:
        return "Fluctuating",     "🟡"
    else:
        return "Highly Unstable", "🔴"
